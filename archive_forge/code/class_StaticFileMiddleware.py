import os
import mimetypes
from datetime import datetime
from time import gmtime
class StaticFileMiddleware(object):
    """A WSGI middleware that provides static content for development
    environments.

    Currently the middleware does not support non ASCII filenames.  If the
    encoding on the file system happens to be the encoding of the URI it may
    work but this could also be by accident.  We strongly suggest using ASCII
    only file names for static files.

    The middleware will guess the mimetype using the Python `mimetype`
    module.  If it's unable to figure out the charset it will fall back
    to `fallback_mimetype`.

    :param app: the application to wrap.  If you don't want to wrap an
                application you can pass it :exc:`NotFound`.
    :param directory: the directory to serve up.
    :param fallback_mimetype: the fallback mimetype for unknown files.
    """

    def __init__(self, app, directory, fallback_mimetype='text/plain'):
        self.app = app
        self.loader = self.get_directory_loader(directory)
        self.fallback_mimetype = fallback_mimetype

    def _opener(self, filename):
        return lambda: (open(filename, 'rb'), datetime.utcfromtimestamp(os.path.getmtime(filename)), int(os.path.getsize(filename)))

    def get_directory_loader(self, directory):

        def loader(path):
            path = path or directory
            if path is not None:
                path = os.path.join(directory, path)
            if os.path.isfile(path):
                return (os.path.basename(path), self._opener(path))
            return (None, None)
        return loader

    def __call__(self, environ, start_response):
        cleaned_path = environ.get('PATH_INFO', '').strip('/')
        for sep in (os.sep, os.altsep):
            if sep and sep != '/':
                cleaned_path = cleaned_path.replace(sep, '/')
        path = '/'.join([''] + [x for x in cleaned_path.split('/') if x and x != '..'])
        real_filename, file_loader = self.loader(path[1:])
        if file_loader is None:
            return self.app(environ, start_response)
        guessed_type = mimetypes.guess_type(real_filename)
        mime_type = guessed_type[0] or self.fallback_mimetype
        f, mtime, file_size = file_loader()
        headers = [('Date', http_date())]
        headers.append(('Cache-Control', 'public'))
        headers.extend((('Content-Type', mime_type), ('Content-Length', str(file_size)), ('Last-Modified', http_date(mtime))))
        start_response('200 OK', headers)
        return wrap_file(environ, f)