import os
import sys
import time
import warnings
import contextlib
import portend
class ServerAdapter(object):
    """Adapter for an HTTP server.

    If you need to start more than one HTTP server (to serve on multiple
    ports, or protocols, etc.), you can manually register each one and then
    start them all with bus.start::

        s1 = ServerAdapter(bus, MyWSGIServer(host='0.0.0.0', port=80))
        s2 = ServerAdapter(bus, another.HTTPServer(host='127.0.0.1', SSL=True))
        s1.subscribe()
        s2.subscribe()
        bus.start()
    """

    def __init__(self, bus, httpserver=None, bind_addr=None):
        self.bus = bus
        self.httpserver = httpserver
        self.bind_addr = bind_addr
        self.interrupt = None
        self.running = False

    def subscribe(self):
        self.bus.subscribe('start', self.start)
        self.bus.subscribe('stop', self.stop)

    def unsubscribe(self):
        self.bus.unsubscribe('start', self.start)
        self.bus.unsubscribe('stop', self.stop)

    def start(self):
        """Start the HTTP server."""
        if self.running:
            self.bus.log('Already serving on %s' % self.description)
            return
        self.interrupt = None
        if not self.httpserver:
            raise ValueError('No HTTP server has been created.')
        if not os.environ.get('LISTEN_PID', None):
            if isinstance(self.bind_addr, tuple):
                portend.free(*self.bind_addr, timeout=Timeouts.free)
        import threading
        t = threading.Thread(target=self._start_http_thread)
        t.name = 'HTTPServer ' + t.name
        t.start()
        self.wait()
        self.running = True
        self.bus.log('Serving on %s' % self.description)
    start.priority = 75

    @property
    def description(self):
        """
        A description about where this server is bound.
        """
        if self.bind_addr is None:
            on_what = 'unknown interface (dynamic?)'
        elif isinstance(self.bind_addr, tuple):
            on_what = self._get_base()
        else:
            on_what = 'socket file: %s' % self.bind_addr
        return on_what

    def _get_base(self):
        if not self.httpserver:
            return ''
        host, port = self.bound_addr
        if getattr(self.httpserver, 'ssl_adapter', None):
            scheme = 'https'
            if port != 443:
                host += ':%s' % port
        else:
            scheme = 'http'
            if port != 80:
                host += ':%s' % port
        return '%s://%s' % (scheme, host)

    def _start_http_thread(self):
        """HTTP servers MUST be running in new threads, so that the
        main thread persists to receive KeyboardInterrupt's. If an
        exception is raised in the httpserver's thread then it's
        trapped here, and the bus (and therefore our httpserver)
        are shut down.
        """
        try:
            self.httpserver.start()
        except KeyboardInterrupt:
            self.bus.log('<Ctrl-C> hit: shutting down HTTP server')
            self.interrupt = sys.exc_info()[1]
            self.bus.exit()
        except SystemExit:
            self.bus.log('SystemExit raised: shutting down HTTP server')
            self.interrupt = sys.exc_info()[1]
            self.bus.exit()
            raise
        except Exception:
            self.interrupt = sys.exc_info()[1]
            self.bus.log('Error in HTTP server: shutting down', traceback=True, level=40)
            self.bus.exit()
            raise

    def wait(self):
        """Wait until the HTTP server is ready to receive requests."""
        while not getattr(self.httpserver, 'ready', False):
            if self.interrupt:
                raise self.interrupt
            time.sleep(0.1)
        if os.environ.get('LISTEN_PID', None):
            return
        if not isinstance(self.bind_addr, tuple):
            return
        with _safe_wait(*self.bound_addr):
            portend.occupied(*self.bound_addr, timeout=Timeouts.occupied)

    @property
    def bound_addr(self):
        """
        The bind address, or if it's an ephemeral port and the
        socket has been bound, return the actual port bound.
        """
        host, port = self.bind_addr
        if port == 0 and self.httpserver.socket:
            port = self.httpserver.socket.getsockname()[1]
        return (host, port)

    def stop(self):
        """Stop the HTTP server."""
        if self.running:
            self.httpserver.stop()
            if isinstance(self.bind_addr, tuple):
                portend.free(*self.bound_addr, timeout=Timeouts.free)
            self.running = False
            self.bus.log('HTTP Server %s shut down' % self.httpserver)
        else:
            self.bus.log('HTTP Server %s already shut down' % self.httpserver)
    stop.priority = 25

    def restart(self):
        """Restart the HTTP server."""
        self.stop()
        self.start()