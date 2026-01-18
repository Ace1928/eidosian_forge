import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
class _StatusBasedRedirect(object):
    """
    Deprectated; use StatusBasedForward instead.
    """

    def __init__(self, app, mapper, global_conf=None, **kw):
        warnings.warn('errordocuments._StatusBasedRedirect has been deprecated; please use errordocuments.StatusBasedForward', DeprecationWarning, 2)
        if global_conf is None:
            global_conf = {}
        self.application = app
        self.mapper = mapper
        self.global_conf = global_conf
        self.kw = kw
        self.fallback_template = '\n            <html>\n            <head>\n            <title>Error %(code)s</title>\n            </html>\n            <body>\n            <h1>Error %(code)s</h1>\n            <p>%(message)s</p>\n            <hr>\n            <p>\n                Additionally an error occurred trying to produce an\n                error document.  A description of the error was logged\n                to <tt>wsgi.errors</tt>.\n            </p>\n            </body>\n            </html>\n        '

    def __call__(self, environ, start_response):
        url = []
        code_message = []
        try:

            def change_response(status, headers, exc_info=None):
                new_url = None
                parts = status.split(' ')
                try:
                    code = int(parts[0])
                except (ValueError, TypeError):
                    raise Exception('_StatusBasedRedirect middleware received an invalid status code %s' % repr(parts[0]))
                message = ' '.join(parts[1:])
                new_url = self.mapper(code, message, environ, self.global_conf, self.kw)
                if not (new_url == None or isinstance(new_url, str)):
                    raise TypeError('Expected the url to internally redirect to in the _StatusBasedRedirect error_mapperto be a string or None, not %s' % repr(new_url))
                if new_url:
                    url.append(new_url)
                code_message.append([code, message])
                return start_response(status, headers, exc_info)
            app_iter = self.application(environ, change_response)
        except:
            try:
                import sys
                error = str(sys.exc_info()[1])
            except:
                error = ''
            try:
                code, message = code_message[0]
            except:
                code, message = ['', '']
            environ['wsgi.errors'].write('Error occurred in _StatusBasedRedirect intercepting the response: ' + str(error))
            return [self.fallback_template % {'message': message, 'code': code}]
        else:
            if url:
                url_ = url[0]
                new_environ = {}
                for k, v in environ.items():
                    if k != 'QUERY_STRING':
                        new_environ['QUERY_STRING'] = urlparse.urlparse(url_)[4]
                    else:
                        new_environ[k] = v

                class InvalidForward(Exception):
                    pass

                def eat_start_response(status, headers, exc_info=None):
                    """
                    We don't want start_response to do anything since it
                    has already been called
                    """
                    if status[:3] != '200':
                        raise InvalidForward("The URL %s to internally forward to in order to create an error document did not return a '200' status code." % url_)
                forward = environ['paste.recursive.forward']
                old_start_response = forward.start_response
                forward.start_response = eat_start_response
                try:
                    app_iter = forward(url_, new_environ)
                except InvalidForward:
                    code, message = code_message[0]
                    environ['wsgi.errors'].write('Error occurred in _StatusBasedRedirect redirecting to new URL: ' + str(url[0]))
                    return [self.fallback_template % {'message': message, 'code': code}]
                else:
                    forward.start_response = old_start_response
                    return app_iter
            else:
                return app_iter