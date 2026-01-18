from __future__ import annotations
import os
import threading
import typing as t
import webbrowser
from tornado import gen, httpserver, ioloop, log, web
from tornado.httpclient import AsyncHTTPClient
from traitlets import Bool, Int, Unicode
from .base import PostProcessorBase
class ServePostProcessor(PostProcessorBase):
    """Post processor designed to serve files

    Proxies reveal.js requests to a CDN if no local reveal.js is present
    """
    open_in_browser = Bool(True, help='Should the browser be opened automatically?').tag(config=True)
    browser = Unicode('', help='Specify what browser should be used to open slides. See\n                      https://docs.python.org/3/library/webbrowser.html#webbrowser.register\n                      to see how keys are mapped to browser executables. If\n                      not specified, the default browser will be determined\n                      by the `webbrowser`\n                      standard library module, which allows setting of the BROWSER\n                      environment variable to override it.\n                      ').tag(config=True)
    reveal_cdn = Unicode('https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.5.0', help='URL for reveal.js CDN.').tag(config=True)
    reveal_prefix = Unicode('reveal.js', help='URL prefix for reveal.js').tag(config=True)
    ip = Unicode('127.0.0.1', help='The IP address to listen on.').tag(config=True)
    port = Int(8000, help='port for the server to listen on.').tag(config=True)

    def postprocess(self, input):
        """Serve the build directory with a webserver."""
        dirname, filename = os.path.split(input)
        handlers: list[tuple[t.Any, ...]] = [('/(.+)', web.StaticFileHandler, {'path': dirname}), ('/', web.RedirectHandler, {'url': '/%s' % filename})]
        if '://' in self.reveal_prefix or self.reveal_prefix.startswith('//'):
            pass
        elif os.path.isdir(os.path.join(dirname, self.reveal_prefix)):
            self.log.info('Serving local %s', self.reveal_prefix)
        else:
            self.log.info('Redirecting %s requests to %s', self.reveal_prefix, self.reveal_cdn)
            handlers.insert(0, ('/(%s)/(.*)' % self.reveal_prefix, ProxyHandler))
        app = web.Application(handlers, cdn=self.reveal_cdn, client=AsyncHTTPClient())
        log.app_log = self.log
        http_server = httpserver.HTTPServer(app)
        http_server.listen(self.port, address=self.ip)
        url = 'http://%s:%i/%s' % (self.ip, self.port, filename)
        print('Serving your slides at %s' % url)
        print('Use Control-C to stop this server')
        if self.open_in_browser:
            try:
                browser = webbrowser.get(self.browser or None)
                b = lambda: browser.open(url, new=2)
                threading.Thread(target=b).start()
            except webbrowser.Error as e:
                self.log.warning('No web browser found: %s.', e)
                browser = None
        try:
            ioloop.IOLoop.instance().start()
        except KeyboardInterrupt:
            print('\nInterrupted')