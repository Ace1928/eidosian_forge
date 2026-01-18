import re
from oslo_log import log as logging
from glance.common import wsgi
from glance.i18n import _LI
class GzipMiddleware(wsgi.Middleware):
    re_zip = re.compile('\\bgzip\\b')

    def __init__(self, app):
        LOG.info(_LI('Initialized gzip middleware'))
        super(GzipMiddleware, self).__init__(app)

    def process_response(self, response):
        request = response.request
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if self.re_zip.search(accept_encoding):
            checksum = response.headers.get('Content-MD5')
            content_type = response.headers.get('Content-Type', '')
            lazy = content_type == 'application/octet-stream'
            response.encode_content(lazy=lazy)
            if checksum:
                response.headers['Content-MD5'] = checksum
        return response