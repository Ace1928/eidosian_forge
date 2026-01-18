import base64
import codecs
import collections
import errno
from random import Random
from socket import error as SocketError
import string
import struct
import sys
import time
import zlib
from eventlet import semaphore
from eventlet import wsgi
from eventlet.green import socket
from eventlet.support import get_errno
class WebSocketWSGI:
    """Wraps a websocket handler function in a WSGI application.

    Use it like this::

      @websocket.WebSocketWSGI
      def my_handler(ws):
          from_browser = ws.wait()
          ws.send("from server")

    The single argument to the function will be an instance of
    :class:`WebSocket`.  To close the socket, simply return from the
    function.  Note that the server will log the websocket request at
    the time of closure.

    An optional argument max_frame_length can be given, which will set the
    maximum incoming *uncompressed* payload length of a frame. By default, this
    is set to 8MiB. Note that excessive values here might create a DOS attack
    vector.
    """

    def __init__(self, handler, max_frame_length=DEFAULT_MAX_FRAME_LENGTH):
        self.handler = handler
        self.protocol_version = None
        self.support_legacy_versions = True
        self.supported_protocols = []
        self.origin_checker = None
        self.max_frame_length = max_frame_length

    @classmethod
    def configured(cls, handler=None, supported_protocols=None, origin_checker=None, support_legacy_versions=False):

        def decorator(handler):
            inst = cls(handler)
            inst.support_legacy_versions = support_legacy_versions
            inst.origin_checker = origin_checker
            if supported_protocols:
                inst.supported_protocols = supported_protocols
            return inst
        if handler is None:
            return decorator
        return decorator(handler)

    def __call__(self, environ, start_response):
        http_connection_parts = [part.strip() for part in environ.get('HTTP_CONNECTION', '').lower().split(',')]
        if not ('upgrade' in http_connection_parts and environ.get('HTTP_UPGRADE', '').lower() == 'websocket'):
            start_response('400 Bad Request', [('Connection', 'close')])
            return []
        try:
            if 'HTTP_SEC_WEBSOCKET_VERSION' in environ:
                ws = self._handle_hybi_request(environ)
            elif self.support_legacy_versions:
                ws = self._handle_legacy_request(environ)
            else:
                raise BadRequest()
        except BadRequest as e:
            status = e.status
            body = e.body or b''
            headers = e.headers or []
            start_response(status, [('Connection', 'close')] + headers)
            return [body]
        if 'eventlet.set_idle' in environ:
            environ['eventlet.set_idle']()
        try:
            self.handler(ws)
        except OSError as e:
            if get_errno(e) not in ACCEPTABLE_CLIENT_ERRORS:
                raise
        ws._send_closing_frame(True)
        wsgi.WSGI_LOCAL.already_handled = True
        return []

    def _handle_legacy_request(self, environ):
        if 'eventlet.input' in environ:
            sock = environ['eventlet.input'].get_socket()
        elif 'gunicorn.socket' in environ:
            sock = environ['gunicorn.socket']
        else:
            raise Exception('No eventlet.input or gunicorn.socket present in environ.')
        if 'HTTP_SEC_WEBSOCKET_KEY1' in environ:
            self.protocol_version = 76
            if 'HTTP_SEC_WEBSOCKET_KEY2' not in environ:
                raise BadRequest()
        else:
            self.protocol_version = 75
        if self.protocol_version == 76:
            key1 = self._extract_number(environ['HTTP_SEC_WEBSOCKET_KEY1'])
            key2 = self._extract_number(environ['HTTP_SEC_WEBSOCKET_KEY2'])
            environ['wsgi.input'].content_length = 8
            key3 = environ['wsgi.input'].read(8)
            key = struct.pack('>II', key1, key2) + key3
            response = md5(key).digest()
        scheme = 'ws'
        if environ.get('wsgi.url_scheme') == 'https':
            scheme = 'wss'
        location = '%s://%s%s%s' % (scheme, environ.get('HTTP_HOST'), environ.get('SCRIPT_NAME'), environ.get('PATH_INFO'))
        qs = environ.get('QUERY_STRING')
        if qs is not None:
            location += '?' + qs
        if self.protocol_version == 75:
            handshake_reply = b'HTTP/1.1 101 Web Socket Protocol Handshake\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nWebSocket-Origin: ' + environ.get('HTTP_ORIGIN').encode() + b'\r\nWebSocket-Location: ' + location.encode() + b'\r\n\r\n'
        elif self.protocol_version == 76:
            handshake_reply = b'HTTP/1.1 101 WebSocket Protocol Handshake\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\nSec-WebSocket-Origin: ' + environ.get('HTTP_ORIGIN').encode() + b'\r\nSec-WebSocket-Protocol: ' + environ.get('HTTP_SEC_WEBSOCKET_PROTOCOL', 'default').encode() + b'\r\nSec-WebSocket-Location: ' + location.encode() + b'\r\n\r\n' + response
        else:
            raise ValueError('Unknown WebSocket protocol version.')
        sock.sendall(handshake_reply)
        return WebSocket(sock, environ, self.protocol_version)

    def _parse_extension_header(self, header):
        if header is None:
            return None
        res = {}
        for ext in header.split(','):
            parts = ext.split(';')
            config = {}
            for part in parts[1:]:
                key_val = part.split('=')
                if len(key_val) == 1:
                    config[key_val[0].strip().lower()] = True
                else:
                    config[key_val[0].strip().lower()] = key_val[1].strip().strip('"').lower()
            res.setdefault(parts[0].strip().lower(), []).append(config)
        return res

    def _negotiate_permessage_deflate(self, extensions):
        if not extensions:
            return None
        deflate = extensions.get('permessage-deflate')
        if deflate is None:
            return None
        for config in deflate:
            want_config = {'server_no_context_takeover': config.get('server_no_context_takeover', False), 'client_no_context_takeover': config.get('client_no_context_takeover', False)}
            max_wbits = min(zlib.MAX_WBITS, 15)
            mwb = config.get('server_max_window_bits')
            if mwb is not None:
                if mwb is True:
                    want_config['server_max_window_bits'] = max_wbits
                else:
                    want_config['server_max_window_bits'] = int(config.get('server_max_window_bits', max_wbits))
                    if not 8 <= want_config['server_max_window_bits'] <= 15:
                        continue
            mwb = config.get('client_max_window_bits')
            if mwb is not None:
                if mwb is True:
                    want_config['client_max_window_bits'] = max_wbits
                else:
                    want_config['client_max_window_bits'] = int(config.get('client_max_window_bits', max_wbits))
                    if not 8 <= want_config['client_max_window_bits'] <= 15:
                        continue
            return want_config
        return None

    def _format_extension_header(self, parsed_extensions):
        if not parsed_extensions:
            return None
        parts = []
        for name, config in parsed_extensions.items():
            ext_parts = [name.encode()]
            for key, value in config.items():
                if value is False:
                    pass
                elif value is True:
                    ext_parts.append(key.encode())
                else:
                    ext_parts.append(('%s=%s' % (key, str(value))).encode())
            parts.append(b'; '.join(ext_parts))
        return b', '.join(parts)

    def _handle_hybi_request(self, environ):
        if 'eventlet.input' in environ:
            sock = environ['eventlet.input'].get_socket()
        elif 'gunicorn.socket' in environ:
            sock = environ['gunicorn.socket']
        else:
            raise Exception('No eventlet.input or gunicorn.socket present in environ.')
        hybi_version = environ['HTTP_SEC_WEBSOCKET_VERSION']
        if hybi_version not in ('8', '13'):
            raise BadRequest(status='426 Upgrade Required', headers=[('Sec-WebSocket-Version', '8, 13')])
        self.protocol_version = int(hybi_version)
        if 'HTTP_SEC_WEBSOCKET_KEY' not in environ:
            raise BadRequest()
        origin = environ.get('HTTP_ORIGIN', environ.get('HTTP_SEC_WEBSOCKET_ORIGIN', '') if self.protocol_version <= 8 else '')
        if self.origin_checker is not None:
            if not self.origin_checker(environ.get('HTTP_HOST'), origin):
                raise BadRequest(status='403 Forbidden')
        protocols = environ.get('HTTP_SEC_WEBSOCKET_PROTOCOL', None)
        negotiated_protocol = None
        if protocols:
            for p in (i.strip() for i in protocols.split(',')):
                if p in self.supported_protocols:
                    negotiated_protocol = p
                    break
        key = environ['HTTP_SEC_WEBSOCKET_KEY']
        response = base64.b64encode(sha1(key.encode() + PROTOCOL_GUID).digest())
        handshake_reply = [b'HTTP/1.1 101 Switching Protocols', b'Upgrade: websocket', b'Connection: Upgrade', b'Sec-WebSocket-Accept: ' + response]
        if negotiated_protocol:
            handshake_reply.append(b'Sec-WebSocket-Protocol: ' + negotiated_protocol.encode())
        parsed_extensions = {}
        extensions = self._parse_extension_header(environ.get('HTTP_SEC_WEBSOCKET_EXTENSIONS'))
        deflate = self._negotiate_permessage_deflate(extensions)
        if deflate is not None:
            parsed_extensions['permessage-deflate'] = deflate
        formatted_ext = self._format_extension_header(parsed_extensions)
        if formatted_ext is not None:
            handshake_reply.append(b'Sec-WebSocket-Extensions: ' + formatted_ext)
        sock.sendall(b'\r\n'.join(handshake_reply) + b'\r\n\r\n')
        return RFC6455WebSocket(sock, environ, self.protocol_version, protocol=negotiated_protocol, extensions=parsed_extensions, max_frame_length=self.max_frame_length)

    def _extract_number(self, value):
        """
        Utility function which, given a string like 'g98sd  5[]221@1', will
        return 9852211. Used to parse the Sec-WebSocket-Key headers.
        """
        out = ''
        spaces = 0
        for char in value:
            if char in string.digits:
                out += char
            elif char == ' ':
                spaces += 1
        return int(out) // spaces