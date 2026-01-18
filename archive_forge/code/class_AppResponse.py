import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class AppResponse(object):
    """WSGI response iterable for CherryPy applications."""

    def __init__(self, environ, start_response, cpapp):
        self.cpapp = cpapp
        try:
            self.environ = environ
            self.run()
            r = _cherrypy.serving.response
            outstatus = r.output_status
            if not isinstance(outstatus, bytes):
                raise TypeError('response.output_status is not a byte string.')
            outheaders = []
            for k, v in r.header_list:
                if not isinstance(k, bytes):
                    tmpl = 'response.header_list key %r is not a byte string.'
                    raise TypeError(tmpl % k)
                if not isinstance(v, bytes):
                    tmpl = 'response.header_list value %r is not a byte string.'
                    raise TypeError(tmpl % v)
                outheaders.append((k, v))
            if True:
                outstatus = outstatus.decode('ISO-8859-1')
                outheaders = [(k.decode('ISO-8859-1'), v.decode('ISO-8859-1')) for k, v in outheaders]
            self.iter_response = iter(r.body)
            self.write = start_response(outstatus, outheaders)
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter_response)

    def close(self):
        """Close and de-reference the current request and response. (Core)"""
        streaming = _cherrypy.serving.response.stream
        self.cpapp.release_serving()
        if streaming and is_closable_iterator(self.iter_response):
            iter_close = self.iter_response.close
            try:
                iter_close()
            except Exception:
                _cherrypy.log(traceback=True, severity=40)

    def run(self):
        """Create a Request object using environ."""
        env = self.environ.get
        local = httputil.Host('', int(env('SERVER_PORT', 80) or -1), env('SERVER_NAME', ''))
        remote = httputil.Host(env('REMOTE_ADDR', ''), int(env('REMOTE_PORT', -1) or -1), env('REMOTE_HOST', ''))
        scheme = env('wsgi.url_scheme')
        sproto = env('ACTUAL_SERVER_PROTOCOL', 'HTTP/1.1')
        request, resp = self.cpapp.get_serving(local, remote, scheme, sproto)
        request.login = env('LOGON_USER') or env('REMOTE_USER') or None
        request.multithread = self.environ['wsgi.multithread']
        request.multiprocess = self.environ['wsgi.multiprocess']
        request.wsgi_environ = self.environ
        request.prev = env('cherrypy.previous_request', None)
        meth = self.environ['REQUEST_METHOD']
        path = httputil.urljoin(self.environ.get('SCRIPT_NAME', ''), self.environ.get('PATH_INFO', ''))
        qs = self.environ.get('QUERY_STRING', '')
        path, qs = self.recode_path_qs(path, qs) or (path, qs)
        rproto = self.environ.get('SERVER_PROTOCOL')
        headers = self.translate_headers(self.environ)
        rfile = self.environ['wsgi.input']
        request.run(meth, path, qs, rproto, headers, rfile)
    headerNames = {'HTTP_CGI_AUTHORIZATION': 'Authorization', 'CONTENT_LENGTH': 'Content-Length', 'CONTENT_TYPE': 'Content-Type', 'REMOTE_HOST': 'Remote-Host', 'REMOTE_ADDR': 'Remote-Addr'}

    def recode_path_qs(self, path, qs):
        old_enc = self.environ.get('wsgi.url_encoding', 'ISO-8859-1')
        new_enc = self.cpapp.find_config(self.environ.get('PATH_INFO', ''), 'request.uri_encoding', 'utf-8')
        if new_enc.lower() == old_enc.lower():
            return
        try:
            return (path.encode(old_enc).decode(new_enc), qs.encode(old_enc).decode(new_enc))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def translate_headers(self, environ):
        """Translate CGI-environ header names to HTTP header names."""
        for cgiName in environ:
            if cgiName in self.headerNames:
                yield (self.headerNames[cgiName], environ[cgiName])
            elif cgiName[:5] == 'HTTP_':
                translatedHeader = cgiName[5:].replace('_', '-')
                yield (translatedHeader, environ[cgiName])