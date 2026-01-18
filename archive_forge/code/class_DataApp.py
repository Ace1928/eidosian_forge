import os, time, mimetypes, zipfile, tarfile
from paste.httpexceptions import (
from paste.httpheaders import (
class DataApp(object):
    """
    Returns an application that will send content in a single chunk,
    this application has support for setting cache-control and for
    responding to conditional (or HEAD) requests.

    Constructor Arguments:

        ``content``     the content being sent to the client

        ``headers``     the headers to send /w the response

        The remaining ``kwargs`` correspond to headers, where the
        underscore is replaced with a dash.  These values are only
        added to the headers if they are not already provided; thus,
        they can be used for default values.  Examples include, but
        are not limited to:

            ``content_type``
            ``content_encoding``
            ``content_location``

    ``cache_control()``

        This method provides validated construction of the ``Cache-Control``
        header as well as providing for automated filling out of the
        ``EXPIRES`` header for HTTP/1.0 clients.

    ``set_content()``

        This method provides a mechanism to set the content after the
        application has been constructed.  This method does things
        like changing ``Last-Modified`` and ``Content-Length`` headers.

    """
    allowed_methods = ('GET', 'HEAD')

    def __init__(self, content, headers=None, allowed_methods=None, **kwargs):
        assert isinstance(headers, (type(None), list))
        self.expires = None
        self.content = None
        self.content_length = None
        self.last_modified = 0
        if allowed_methods is not None:
            self.allowed_methods = allowed_methods
        self.headers = headers or []
        for k, v in kwargs.items():
            header = get_header(k)
            header.update(self.headers, v)
        ACCEPT_RANGES.update(self.headers, bytes=True)
        if not CONTENT_TYPE(self.headers):
            CONTENT_TYPE.update(self.headers)
        if content is not None:
            self.set_content(content)

    def cache_control(self, **kwargs):
        self.expires = CACHE_CONTROL.apply(self.headers, **kwargs) or None
        return self

    def set_content(self, content, last_modified=None):
        assert content is not None
        if last_modified is None:
            self.last_modified = time.time()
        else:
            self.last_modified = last_modified
        self.content = content
        self.content_length = len(content)
        LAST_MODIFIED.update(self.headers, time=self.last_modified)
        return self

    def content_disposition(self, **kwargs):
        CONTENT_DISPOSITION.apply(self.headers, **kwargs)
        return self

    def __call__(self, environ, start_response):
        method = environ['REQUEST_METHOD'].upper()
        if method not in self.allowed_methods:
            exc = HTTPMethodNotAllowed('You cannot %s a file' % method, headers=[('Allow', ','.join(self.allowed_methods))])
            return exc(environ, start_response)
        return self.get(environ, start_response)

    def calculate_etag(self):
        return '"%s-%s"' % (self.last_modified, self.content_length)

    def get(self, environ, start_response):
        headers = self.headers[:]
        current_etag = self.calculate_etag()
        ETAG.update(headers, current_etag)
        if self.expires is not None:
            EXPIRES.update(headers, delta=self.expires)
        try:
            client_etags = IF_NONE_MATCH.parse(environ)
            if client_etags:
                for etag in client_etags:
                    if etag == current_etag or etag == '*':
                        for head in list_headers(entity=True):
                            head.delete(headers)
                        start_response('304 Not Modified', headers)
                        return [b'']
        except HTTPBadRequest as exce:
            return exce.wsgi_application(environ, start_response)
        if not client_etags:
            try:
                client_clock = IF_MODIFIED_SINCE.parse(environ)
                if client_clock is not None and client_clock >= int(self.last_modified):
                    for head in list_headers(entity=True):
                        head.delete(headers)
                    start_response('304 Not Modified', headers)
                    return [b'']
            except HTTPBadRequest as exce:
                return exce.wsgi_application(environ, start_response)
        lower, upper = (0, self.content_length - 1)
        range = RANGE.parse(environ)
        if range and 'bytes' == range[0] and (1 == len(range[1])):
            lower, upper = range[1][0]
            upper = upper or self.content_length - 1
            if upper >= self.content_length or lower > upper:
                return HTTPRequestRangeNotSatisfiable('Range request was made beyond the end of the content,\r\nwhich is %s long.\r\n  Range: %s\r\n' % (self.content_length, RANGE(environ))).wsgi_application(environ, start_response)
        content_length = upper - lower + 1
        CONTENT_RANGE.update(headers, first_byte=lower, last_byte=upper, total_length=self.content_length)
        CONTENT_LENGTH.update(headers, content_length)
        if range or content_length != self.content_length:
            start_response('206 Partial Content', headers)
        else:
            start_response('200 OK', headers)
        if self.content is not None:
            return [self.content[lower:upper + 1]]
        return (lower, content_length)