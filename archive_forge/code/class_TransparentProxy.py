import http.client as httplib
from urllib import parse as urlparse
from urllib.parse import quote
from paste import httpexceptions
from paste.util.converters import aslist
class TransparentProxy(object):
    """
    A proxy that sends the request just as it was given, including
    respecting HTTP_HOST, wsgi.url_scheme, etc.

    This is a way of translating WSGI requests directly to real HTTP
    requests.  All information goes in the environment; modify it to
    modify the way the request is made.

    If you specify ``force_host`` (and optionally ``force_scheme``)
    then HTTP_HOST won't be used to determine where to connect to;
    instead a specific host will be connected to, but the ``Host``
    header in the request will remain intact.
    """

    def __init__(self, force_host=None, force_scheme='http'):
        self.force_host = force_host
        self.force_scheme = force_scheme

    def __repr__(self):
        return '<%s %s force_host=%r force_scheme=%r>' % (self.__class__.__name__, hex(id(self)), self.force_host, self.force_scheme)

    def __call__(self, environ, start_response):
        scheme = environ['wsgi.url_scheme']
        if self.force_host is None:
            conn_scheme = scheme
        else:
            conn_scheme = self.force_scheme
        if conn_scheme == 'http':
            ConnClass = httplib.HTTPConnection
        elif conn_scheme == 'https':
            ConnClass = httplib.HTTPSConnection
        else:
            raise ValueError('Unknown scheme %r' % scheme)
        if 'HTTP_HOST' not in environ:
            raise ValueError('WSGI environ must contain an HTTP_HOST key')
        host = environ['HTTP_HOST']
        if self.force_host is None:
            conn_host = host
        else:
            conn_host = self.force_host
        conn = ConnClass(conn_host)
        headers = {}
        for key, value in environ.items():
            if key.startswith('HTTP_'):
                key = key[5:].lower().replace('_', '-')
                headers[key] = value
        headers['host'] = host
        if 'REMOTE_ADDR' in environ and 'HTTP_X_FORWARDED_FOR' not in environ:
            headers['x-forwarded-for'] = environ['REMOTE_ADDR']
        if environ.get('CONTENT_TYPE'):
            headers['content-type'] = environ['CONTENT_TYPE']
        if environ.get('CONTENT_LENGTH'):
            length = int(environ['CONTENT_LENGTH'])
            body = environ['wsgi.input'].read(length)
            if length == -1:
                environ['CONTENT_LENGTH'] = str(len(body))
        elif 'CONTENT_LENGTH' not in environ:
            body = ''
            length = 0
        else:
            body = ''
            length = 0
        path = environ.get('SCRIPT_NAME', '') + environ.get('PATH_INFO', '')
        path = quote(path)
        if 'QUERY_STRING' in environ:
            path += '?' + environ['QUERY_STRING']
        conn.request(environ['REQUEST_METHOD'], path, body, headers)
        res = conn.getresponse()
        headers_out = parse_headers(res.msg)
        status = '%s %s' % (res.status, res.reason)
        start_response(status, headers_out)
        length = res.getheader('content-length')
        if length is not None:
            body = res.read(int(length))
        else:
            body = res.read()
        conn.close()
        return [body]