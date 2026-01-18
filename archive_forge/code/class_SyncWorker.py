from datetime import datetime
import errno
import os
import select
import socket
import ssl
import sys
from gunicorn import http
from gunicorn.http import wsgi
from gunicorn import sock
from gunicorn import util
from gunicorn.workers import base
class SyncWorker(base.Worker):

    def accept(self, listener):
        client, addr = listener.accept()
        client.setblocking(1)
        util.close_on_exec(client)
        self.handle(listener, client, addr)

    def wait(self, timeout):
        try:
            self.notify()
            ret = select.select(self.wait_fds, [], [], timeout)
            if ret[0]:
                if self.PIPE[0] in ret[0]:
                    os.read(self.PIPE[0], 1)
                return ret[0]
        except select.error as e:
            if e.args[0] == errno.EINTR:
                return self.sockets
            if e.args[0] == errno.EBADF:
                if self.nr < 0:
                    return self.sockets
                else:
                    raise StopWaiting
            raise

    def is_parent_alive(self):
        if self.ppid != os.getppid():
            self.log.info('Parent changed, shutting down: %s', self)
            return False
        return True

    def run_for_one(self, timeout):
        listener = self.sockets[0]
        while self.alive:
            self.notify()
            try:
                self.accept(listener)
                continue
            except EnvironmentError as e:
                if e.errno not in (errno.EAGAIN, errno.ECONNABORTED, errno.EWOULDBLOCK):
                    raise
            if not self.is_parent_alive():
                return
            try:
                self.wait(timeout)
            except StopWaiting:
                return

    def run_for_multiple(self, timeout):
        while self.alive:
            self.notify()
            try:
                ready = self.wait(timeout)
            except StopWaiting:
                return
            if ready is not None:
                for listener in ready:
                    if listener == self.PIPE[0]:
                        continue
                    try:
                        self.accept(listener)
                    except EnvironmentError as e:
                        if e.errno not in (errno.EAGAIN, errno.ECONNABORTED, errno.EWOULDBLOCK):
                            raise
            if not self.is_parent_alive():
                return

    def run(self):
        timeout = self.timeout or 0.5
        for s in self.sockets:
            s.setblocking(0)
        if len(self.sockets) > 1:
            self.run_for_multiple(timeout)
        else:
            self.run_for_one(timeout)

    def handle(self, listener, client, addr):
        req = None
        try:
            if self.cfg.is_ssl:
                client = sock.ssl_wrap_socket(client, self.cfg)
            parser = http.RequestParser(self.cfg, client, addr)
            req = next(parser)
            self.handle_request(listener, req, client, addr)
        except http.errors.NoMoreData as e:
            self.log.debug('Ignored premature client disconnection. %s', e)
        except StopIteration as e:
            self.log.debug('Closing connection. %s', e)
        except ssl.SSLError as e:
            if e.args[0] == ssl.SSL_ERROR_EOF:
                self.log.debug('ssl connection closed')
                client.close()
            else:
                self.log.debug('Error processing SSL request.')
                self.handle_error(req, client, addr, e)
        except EnvironmentError as e:
            if e.errno not in (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN):
                self.log.exception('Socket error processing request.')
            elif e.errno == errno.ECONNRESET:
                self.log.debug('Ignoring connection reset')
            elif e.errno == errno.ENOTCONN:
                self.log.debug('Ignoring socket not connected')
            else:
                self.log.debug('Ignoring EPIPE')
        except Exception as e:
            self.handle_error(req, client, addr, e)
        finally:
            util.close(client)

    def handle_request(self, listener, req, client, addr):
        environ = {}
        resp = None
        try:
            self.cfg.pre_request(self, req)
            request_start = datetime.now()
            resp, environ = wsgi.create(req, client, addr, listener.getsockname(), self.cfg)
            resp.force_close()
            self.nr += 1
            if self.nr >= self.max_requests:
                self.log.info('Autorestarting worker after current request.')
                self.alive = False
            respiter = self.wsgi(environ, resp.start_response)
            try:
                if isinstance(respiter, environ['wsgi.file_wrapper']):
                    resp.write_file(respiter)
                else:
                    for item in respiter:
                        resp.write(item)
                resp.close()
            finally:
                request_time = datetime.now() - request_start
                self.log.access(resp, req, environ, request_time)
                if hasattr(respiter, 'close'):
                    respiter.close()
        except EnvironmentError:
            util.reraise(*sys.exc_info())
        except Exception:
            if resp and resp.headers_sent:
                self.log.exception('Error handling request')
                try:
                    client.shutdown(socket.SHUT_RDWR)
                    client.close()
                except EnvironmentError:
                    pass
                raise StopIteration()
            raise
        finally:
            try:
                self.cfg.post_request(self, req, environ, resp)
            except Exception:
                self.log.exception('Exception in post_request hook')