import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
class StaticURLParser(object):
    """
    Like ``URLParser`` but only serves static files.

    ``cache_max_age``:
      integer specifies Cache-Control max_age in seconds
    """

    def __init__(self, directory, root_directory=None, cache_max_age=None):
        self.directory = self.normpath(directory)
        self.root_directory = self.normpath(root_directory or directory)
        self.cache_max_age = cache_max_age

    def normpath(path):
        return os.path.normcase(os.path.abspath(path))
    normpath = staticmethod(normpath)

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        if not path_info:
            return self.add_slash(environ, start_response)
        if path_info == '/':
            filename = 'index.html'
        else:
            filename = request.path_info_pop(environ)
        full = self.normpath(os.path.join(self.directory, filename))
        if not full.startswith(self.root_directory):
            return self.not_found(environ, start_response)
        if not os.path.exists(full):
            return self.not_found(environ, start_response)
        if os.path.isdir(full):
            return self.__class__(full, root_directory=self.root_directory, cache_max_age=self.cache_max_age)(environ, start_response)
        if environ.get('PATH_INFO') and environ.get('PATH_INFO') != '/':
            return self.error_extra_path(environ, start_response)
        if_none_match = environ.get('HTTP_IF_NONE_MATCH')
        if if_none_match:
            mytime = os.stat(full).st_mtime
            if str(mytime) == if_none_match:
                headers = []
                ETAG.update(headers, mytime)
                start_response('304 Not Modified', headers)
                return ['']
        fa = self.make_app(full)
        if self.cache_max_age:
            fa.cache_control(max_age=self.cache_max_age)
        return fa(environ, start_response)

    def make_app(self, filename):
        return fileapp.FileApp(filename)

    def add_slash(self, environ, start_response):
        """
        This happens when you try to get to a directory
        without a trailing /
        """
        url = request.construct_url(environ, with_query_string=False)
        url += '/'
        if environ.get('QUERY_STRING'):
            url += '?' + environ['QUERY_STRING']
        exc = httpexceptions.HTTPMovedPermanently('The resource has moved to %s - you should be redirected automatically.' % url, headers=[('location', url)])
        return exc.wsgi_application(environ, start_response)

    def not_found(self, environ, start_response, debug_message=None):
        exc = httpexceptions.HTTPNotFound('The resource at %s could not be found' % request.construct_url(environ), comment='SCRIPT_NAME=%r; PATH_INFO=%r; looking in %r; debug: %s' % (environ.get('SCRIPT_NAME'), environ.get('PATH_INFO'), self.directory, debug_message or '(none)'))
        return exc.wsgi_application(environ, start_response)

    def error_extra_path(self, environ, start_response):
        exc = httpexceptions.HTTPNotFound('The trailing path %r is not allowed' % environ['PATH_INFO'])
        return exc.wsgi_application(environ, start_response)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.directory)