import mimetypes
import os
from webob import exc
from webob.dec import wsgify
from webob.response import Response
class FileApp(object):
    """An application that will send the file at the given filename.

    Adds a mime type based on `mimetypes.guess_type()`.
    """

    def __init__(self, filename, **kw):
        self.filename = filename
        content_type, content_encoding = mimetypes.guess_type(filename)
        kw.setdefault('content_type', content_type)
        kw.setdefault('content_encoding', content_encoding)
        kw.setdefault('accept_ranges', 'bytes')
        self.kw = kw
        self._open = open

    @wsgify
    def __call__(self, req):
        if req.method not in ('GET', 'HEAD'):
            return exc.HTTPMethodNotAllowed('You cannot %s a file' % req.method)
        try:
            stat = os.stat(self.filename)
        except (IOError, OSError) as e:
            msg = "Can't open %r: %s" % (self.filename, e)
            return exc.HTTPNotFound(comment=msg)
        try:
            file = self._open(self.filename, 'rb')
        except (IOError, OSError) as e:
            msg = 'You are not permitted to view this file (%s)' % e
            return exc.HTTPForbidden(msg)
        if 'wsgi.file_wrapper' in req.environ:
            app_iter = req.environ['wsgi.file_wrapper'](file, BLOCK_SIZE)
        else:
            app_iter = FileIter(file)
        return Response(app_iter=app_iter, content_length=stat.st_size, last_modified=stat.st_mtime, **self.kw).conditional_response_app