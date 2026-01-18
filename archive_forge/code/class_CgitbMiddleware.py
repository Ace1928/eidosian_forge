import cgitb
from io import StringIO
import sys
from paste.util import converters
class CgitbMiddleware(object):

    def __init__(self, app, global_conf=None, display=NoDefault, logdir=None, context=5, format='html'):
        self.app = app
        if global_conf is None:
            global_conf = {}
        if display is NoDefault:
            display = global_conf.get('debug')
        if isinstance(display, str):
            display = converters.asbool(display)
        self.display = display
        self.logdir = logdir
        self.context = int(context)
        self.format = format

    def __call__(self, environ, start_response):
        try:
            app_iter = self.app(environ, start_response)
            return self.catching_iter(app_iter, environ)
        except:
            exc_info = sys.exc_info()
            start_response('500 Internal Server Error', [('content-type', 'text/html')], exc_info)
            response = self.exception_handler(exc_info, environ)
            response = response.encode('utf8')
            return [response]

    def catching_iter(self, app_iter, environ):
        if not app_iter:
            return
        error_on_close = False
        try:
            for v in app_iter:
                yield v
            if hasattr(app_iter, 'close'):
                error_on_close = True
                app_iter.close()
        except:
            response = self.exception_handler(sys.exc_info(), environ)
            if not error_on_close and hasattr(app_iter, 'close'):
                try:
                    app_iter.close()
                except:
                    close_response = self.exception_handler(sys.exc_info(), environ)
                    response += '<hr noshade>Error in .close():<br>%s' % close_response
            response = response.encode('utf8')
            yield response

    def exception_handler(self, exc_info, environ):
        dummy_file = StringIO()
        hook = cgitb.Hook(file=dummy_file, display=self.display, logdir=self.logdir, context=self.context, format=self.format)
        hook(*exc_info)
        return dummy_file.getvalue()