import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
class HttpProtocol(BaseHTTPServer.BaseHTTPRequestHandler):
    """This class is used to handle the HTTP requests that arrive
    at the server.

    The handler will parse the request and the headers, then call a method
    specific to the request type.

    :param conn_state: The given connection status.
    :param server: The server accessible by the request handler.
    """
    protocol_version = 'HTTP/1.1'
    minimum_chunk_size = MINIMUM_CHUNK_SIZE
    capitalize_response_headers = True
    reject_bad_requests = True
    wbufsize = 16 << 10

    def __init__(self, conn_state, server):
        self.request = conn_state[1]
        self.client_address = conn_state[0]
        self.conn_state = conn_state
        self.server = server
        self.setup()
        try:
            self.handle()
        finally:
            self.finish()

    def setup(self):
        conn = self.connection = self.request
        if getattr(socket, 'TCP_QUICKACK', None):
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, True)
            except OSError:
                pass
        try:
            self.rfile = conn.makefile('rb', self.rbufsize)
            self.wfile = conn.makefile('wb', self.wbufsize)
        except (AttributeError, NotImplementedError):
            if hasattr(conn, 'send') and hasattr(conn, 'recv'):
                self.rfile = socket._fileobject(conn, 'rb', self.rbufsize)
                self.wfile = socket._fileobject(conn, 'wb', self.wbufsize)
            else:
                raise NotImplementedError("eventlet.wsgi doesn't support sockets of type {}".format(type(conn)))

    def handle(self):
        self.close_connection = True
        while True:
            self.handle_one_request()
            if self.conn_state[2] == STATE_CLOSE:
                self.close_connection = 1
            else:
                self.conn_state[2] = STATE_IDLE
            if self.close_connection:
                break

    def _read_request_line(self):
        if self.rfile.closed:
            self.close_connection = 1
            return ''
        try:
            sock = self.connection
            if self.server.keepalive and (not isinstance(self.server.keepalive, bool)):
                sock.settimeout(self.server.keepalive)
            line = self.rfile.readline(self.server.url_length_limit)
            sock.settimeout(self.server.socket_timeout)
            return line
        except greenio.SSL.ZeroReturnError:
            pass
        except OSError as e:
            last_errno = support.get_errno(e)
            if last_errno in BROKEN_SOCK:
                self.server.log.debug('({}) connection reset by peer {!r}'.format(self.server.pid, self.client_address))
            elif last_errno not in BAD_SOCK:
                raise
        return ''

    def handle_one_request(self):
        if self.server.max_http_version:
            self.protocol_version = self.server.max_http_version
        self.raw_requestline = self._read_request_line()
        self.conn_state[2] = STATE_REQUEST
        if not self.raw_requestline:
            self.close_connection = 1
            return
        if len(self.raw_requestline) >= self.server.url_length_limit:
            self.wfile.write(RESPONSE_414)
            self.close_connection = 1
            return
        orig_rfile = self.rfile
        try:
            self.rfile = FileObjectForHeaders(self.rfile)
            if not self.parse_request():
                return
        except HeaderLineTooLong:
            self.wfile.write(b'HTTP/1.0 400 Header Line Too Long\r\nConnection: close\r\nContent-length: 0\r\n\r\n')
            self.close_connection = 1
            return
        except HeadersTooLarge:
            self.wfile.write(b'HTTP/1.0 400 Headers Too Large\r\nConnection: close\r\nContent-length: 0\r\n\r\n')
            self.close_connection = 1
            return
        finally:
            self.rfile = orig_rfile
        content_length = self.headers.get('content-length')
        transfer_encoding = self.headers.get('transfer-encoding')
        if content_length is not None:
            try:
                if int(content_length) < 0:
                    raise ValueError
            except ValueError:
                self.wfile.write(b'HTTP/1.0 400 Bad Request\r\nConnection: close\r\nContent-length: 0\r\n\r\n')
                self.close_connection = 1
                return
            if transfer_encoding is not None:
                if self.reject_bad_requests:
                    msg = b'Content-Length and Transfer-Encoding are not allowed together\n'
                    self.wfile.write(b'HTTP/1.0 400 Bad Request\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s' % (len(msg), msg))
                    self.close_connection = 1
                    return
        self.environ = self.get_environ()
        self.application = self.server.app
        try:
            self.server.outstanding_requests += 1
            try:
                self.handle_one_response()
            except OSError as e:
                if support.get_errno(e) not in BROKEN_SOCK:
                    raise
        finally:
            self.server.outstanding_requests -= 1

    def handle_one_response(self):
        start = time.time()
        headers_set = []
        headers_sent = []
        request_input = self.environ['eventlet.input']
        request_input.headers_sent = headers_sent
        wfile = self.wfile
        result = None
        use_chunked = [False]
        length = [0]
        status_code = [200]

        def write(data):
            towrite = []
            if not headers_set:
                raise AssertionError('write() before start_response()')
            elif not headers_sent:
                status, response_headers = headers_set
                headers_sent.append(1)
                header_list = [header[0].lower() for header in response_headers]
                towrite.append(('%s %s\r\n' % (self.protocol_version, status)).encode())
                for header in response_headers:
                    towrite.append(('%s: %s\r\n' % header).encode('latin-1'))
                if 'date' not in header_list:
                    towrite.append(('Date: %s\r\n' % (format_date_time(time.time()),)).encode())
                client_conn = self.headers.get('Connection', '').lower()
                send_keep_alive = False
                if self.close_connection == 0 and self.server.keepalive and (client_conn == 'keep-alive' or (self.request_version == 'HTTP/1.1' and (not client_conn == 'close'))):
                    send_keep_alive = client_conn == 'keep-alive'
                    self.close_connection = 0
                else:
                    self.close_connection = 1
                if 'content-length' not in header_list:
                    if self.request_version == 'HTTP/1.1':
                        use_chunked[0] = True
                        towrite.append(b'Transfer-Encoding: chunked\r\n')
                    elif 'content-length' not in header_list:
                        self.close_connection = 1
                if self.close_connection:
                    towrite.append(b'Connection: close\r\n')
                elif send_keep_alive:
                    towrite.append(b'Connection: keep-alive\r\n')
                    int_timeout = int(self.server.keepalive or 0)
                    if not isinstance(self.server.keepalive, bool) and int_timeout:
                        towrite.append(b'Keep-Alive: timeout=%d\r\n' % int_timeout)
                towrite.append(b'\r\n')
            if use_chunked[0]:
                towrite.append(('%x' % (len(data),)).encode() + b'\r\n' + data + b'\r\n')
            else:
                towrite.append(data)
            wfile.writelines(towrite)
            wfile.flush()
            length[0] = length[0] + sum(map(len, towrite))

        def start_response(status, response_headers, exc_info=None):
            status_code[0] = status.split()[0]
            if exc_info:
                try:
                    if headers_sent:
                        raise exc_info[1].with_traceback(exc_info[2])
                finally:
                    exc_info = None
            if self.capitalize_response_headers:

                def cap(x):
                    return x.encode('latin1').capitalize().decode('latin1')
                response_headers = [('-'.join([cap(x) for x in key.split('-')]), value) for key, value in response_headers]
            headers_set[:] = [status, response_headers]
            return write
        try:
            try:
                WSGI_LOCAL.already_handled = False
                result = self.application(self.environ, start_response)
                if headers_set and (not headers_sent) and hasattr(result, '__len__'):
                    if 'Content-Length' not in [h for h, _v in headers_set[1]]:
                        headers_set[1].append(('Content-Length', str(sum(map(len, result)))))
                    if request_input.should_send_hundred_continue:
                        self.close_connection = 1
                towrite = []
                towrite_size = 0
                just_written_size = 0
                minimum_write_chunk_size = int(self.environ.get('eventlet.minimum_write_chunk_size', self.minimum_chunk_size))
                for data in result:
                    if len(data) == 0:
                        continue
                    if isinstance(data, str):
                        data = data.encode('ascii')
                    towrite.append(data)
                    towrite_size += len(data)
                    if towrite_size >= minimum_write_chunk_size:
                        write(b''.join(towrite))
                        towrite = []
                        just_written_size = towrite_size
                        towrite_size = 0
                if WSGI_LOCAL.already_handled:
                    self.close_connection = 1
                    return
                if towrite:
                    just_written_size = towrite_size
                    write(b''.join(towrite))
                if not headers_sent or (use_chunked[0] and just_written_size):
                    write(b'')
            except (Exception, eventlet.Timeout):
                self.close_connection = 1
                tb = traceback.format_exc()
                self.server.log.info(tb)
                if not headers_sent:
                    err_body = tb.encode() if self.server.debug else b''
                    start_response('500 Internal Server Error', [('Content-type', 'text/plain'), ('Content-length', len(err_body))])
                    write(err_body)
        finally:
            if hasattr(result, 'close'):
                result.close()
            if request_input.should_send_hundred_continue:
                self.close_connection = 1
            if request_input.chunked_input or request_input.position < (request_input.content_length or 0):
                if self.close_connection == 0:
                    try:
                        request_input.discard()
                    except ChunkReadError as e:
                        self.close_connection = 1
                        self.server.log.error(('chunked encoding error while discarding request body.' + ' client={0} request="{1}" error="{2}"').format(self.get_client_address()[0], self.requestline, e))
                    except OSError as e:
                        self.close_connection = 1
                        self.server.log.error(('I/O error while discarding request body.' + ' client={0} request="{1}" error="{2}"').format(self.get_client_address()[0], self.requestline, e))
            finish = time.time()
            for hook, args, kwargs in self.environ['eventlet.posthooks']:
                hook(self.environ, *args, **kwargs)
            if self.server.log_output:
                client_host, client_port = self.get_client_address()
                self.server.log.info(self.server.log_format % {'client_ip': client_host, 'client_port': client_port, 'date_time': self.log_date_time_string(), 'request_line': self.requestline, 'status_code': status_code[0], 'body_length': length[0], 'wall_seconds': finish - start})

    def get_client_address(self):
        host, port = addr_to_host_port(self.client_address)
        if self.server.log_x_forwarded_for:
            forward = self.headers.get('X-Forwarded-For', '').replace(' ', '')
            if forward:
                host = forward + ',' + host
        return (host, port)

    def get_environ(self):
        env = self.server.get_environ()
        env['REQUEST_METHOD'] = self.command
        env['SCRIPT_NAME'] = ''
        pq = self.path.split('?', 1)
        env['RAW_PATH_INFO'] = pq[0]
        env['PATH_INFO'] = urllib.parse.unquote(pq[0], encoding='latin1')
        if len(pq) > 1:
            env['QUERY_STRING'] = pq[1]
        ct = self.headers.get('content-type')
        if ct is None:
            try:
                ct = self.headers.type
            except AttributeError:
                ct = self.headers.get_content_type()
        env['CONTENT_TYPE'] = ct
        length = self.headers.get('content-length')
        if length:
            env['CONTENT_LENGTH'] = length
        env['SERVER_PROTOCOL'] = 'HTTP/1.0'
        sockname = self.request.getsockname()
        server_addr = addr_to_host_port(sockname)
        env['SERVER_NAME'] = server_addr[0]
        env['SERVER_PORT'] = str(server_addr[1])
        client_addr = addr_to_host_port(self.client_address)
        env['REMOTE_ADDR'] = client_addr[0]
        env['REMOTE_PORT'] = str(client_addr[1])
        env['GATEWAY_INTERFACE'] = 'CGI/1.1'
        try:
            headers = self.headers.headers
        except AttributeError:
            headers = self.headers._headers
        else:
            headers = [h.split(':', 1) for h in headers]
        env['headers_raw'] = headers_raw = tuple(((k, v.strip(' \t\n\r')) for k, v in headers))
        for k, v in headers_raw:
            k = k.replace('-', '_').upper()
            if k in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                continue
            envk = 'HTTP_' + k
            if envk in env:
                env[envk] += ',' + v
            else:
                env[envk] = v
        if env.get('HTTP_EXPECT', '').lower() == '100-continue':
            wfile = self.wfile
            wfile_line = b'HTTP/1.1 100 Continue\r\n'
        else:
            wfile = None
            wfile_line = None
        chunked = env.get('HTTP_TRANSFER_ENCODING', '').lower() == 'chunked'
        env['wsgi.input'] = env['eventlet.input'] = Input(self.rfile, length, self.connection, wfile=wfile, wfile_line=wfile_line, chunked_input=chunked)
        env['eventlet.posthooks'] = []

        def set_idle():
            self.conn_state[2] = STATE_IDLE
        env['eventlet.set_idle'] = set_idle
        return env

    def finish(self):
        try:
            BaseHTTPServer.BaseHTTPRequestHandler.finish(self)
        except OSError as e:
            if support.get_errno(e) not in BROKEN_SOCK:
                raise
        greenio.shutdown_safe(self.connection)
        self.connection.close()

    def handle_expect_100(self):
        return True