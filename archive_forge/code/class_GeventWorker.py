import os
import sys
from datetime import datetime
from functools import partial
import time
from gevent.pool import Pool
from gevent.server import StreamServer
from gevent import hub, monkey, socket, pywsgi
import gunicorn
from gunicorn.http.wsgi import base_environ
from gunicorn.sock import ssl_context
from gunicorn.workers.base_async import AsyncWorker
class GeventWorker(AsyncWorker):
    server_class = None
    wsgi_handler = None

    def patch(self):
        monkey.patch_all()
        sockets = []
        for s in self.sockets:
            sockets.append(socket.socket(s.FAMILY, socket.SOCK_STREAM, fileno=s.sock.fileno()))
        self.sockets = sockets

    def notify(self):
        super().notify()
        if self.ppid != os.getppid():
            self.log.info('Parent changed, shutting down: %s', self)
            sys.exit(0)

    def timeout_ctx(self):
        return gevent.Timeout(self.cfg.keepalive, False)

    def run(self):
        servers = []
        ssl_args = {}
        if self.cfg.is_ssl:
            ssl_args = {'ssl_context': ssl_context(self.cfg)}
        for s in self.sockets:
            s.setblocking(1)
            pool = Pool(self.worker_connections)
            if self.server_class is not None:
                environ = base_environ(self.cfg)
                environ.update({'wsgi.multithread': True, 'SERVER_SOFTWARE': VERSION})
                server = self.server_class(s, application=self.wsgi, spawn=pool, log=self.log, handler_class=self.wsgi_handler, environ=environ, **ssl_args)
            else:
                hfun = partial(self.handle, s)
                server = StreamServer(s, handle=hfun, spawn=pool, **ssl_args)
                if self.cfg.workers > 1:
                    server.max_accept = 1
            server.start()
            servers.append(server)
        while self.alive:
            self.notify()
            gevent.sleep(1.0)
        try:
            for server in servers:
                if hasattr(server, 'close'):
                    server.close()
                if hasattr(server, 'kill'):
                    server.kill()
            ts = time.time()
            while time.time() - ts <= self.cfg.graceful_timeout:
                accepting = 0
                for server in servers:
                    if server.pool.free_count() != server.pool.size:
                        accepting += 1
                if not accepting:
                    return
                self.notify()
                gevent.sleep(1.0)
            self.log.warning('Worker graceful timeout (pid:%s)', self.pid)
            for server in servers:
                server.stop(timeout=1)
        except Exception:
            pass

    def handle(self, listener, client, addr):
        client.setblocking(1)
        super().handle(listener, client, addr)

    def handle_request(self, listener_name, req, sock, addr):
        try:
            super().handle_request(listener_name, req, sock, addr)
        except gevent.GreenletExit:
            pass
        except SystemExit:
            pass

    def handle_quit(self, sig, frame):
        gevent.spawn(super().handle_quit, sig, frame)

    def handle_usr1(self, sig, frame):
        gevent.spawn(super().handle_usr1, sig, frame)

    def init_process(self):
        self.patch()
        hub.reinit()
        super().init_process()