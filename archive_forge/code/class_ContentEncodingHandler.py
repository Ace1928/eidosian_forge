import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
class ContentEncodingHandler(BaseHandler):
    """Request and handle HTTP Content-Encoding."""

    def http_request(self, request):
        request.add_header('Accept-Encoding', 'gzip')
        for header in request.headers:
            if header.lower() == 'user-agent':
                request.headers[header] += ' gzip'
        return request
    https_request = http_request

    def http_response(self, req, resp):
        """Handle encodings in the order that they are encountered."""
        encodings = []
        headers = resp.headers
        encoding_header = None
        for header in headers:
            if header.lower() == 'content-encoding':
                encoding_header = header
                for encoding in headers[header].split(','):
                    encoding = encoding.strip()
                    if encoding:
                        encodings.append(encoding)
                break
        if not encodings:
            return resp
        del headers[encoding_header]
        fp = resp
        while encodings and encodings[-1].lower() == 'gzip':
            fp = io.BytesIO(fp.read())
            fp = gzip.GzipFile(fileobj=fp, mode='r')
            encodings.pop()
        if encodings:
            headers[encoding_header] = ', '.join(encodings)
            logger.warning('Unrecognized Content-Encoding: %s', encodings[-1])
        msg = resp.msg
        if sys.version_info >= (2, 6):
            resp = addinfourl_fn(fp, headers, resp.url, resp.code)
        else:
            response_code = resp.code
            resp = addinfourl_fn(fp, headers, resp.url)
            resp.code = response_code
        resp.msg = msg
        return resp
    https_response = http_response