from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def _post_request(self, request, params, parser, body='', headers=None):
    """Make a POST request, optionally with a content body,
           and return the response, optionally as raw text.
        """
    headers = headers or {}
    path = self._sandboxify(request['path'])
    request = self.build_base_http_request('POST', path, None, data=body, params=params, headers=headers, host=self.host)
    try:
        response = self._mexe(request, override_num_retries=None)
    except BotoServerError as bs:
        raise self._response_error_factory(bs.status, bs.reason, bs.body)
    body = response.read()
    boto.log.debug(body)
    if not body:
        boto.log.error('Null body %s' % body)
        raise self._response_error_factory(response.status, response.reason, body)
    if response.status != 200:
        boto.log.error('%s %s' % (response.status, response.reason))
        boto.log.error('%s' % body)
        raise self._response_error_factory(response.status, response.reason, body)
    digest = response.getheader('Content-MD5')
    if digest is not None:
        assert content_md5(body) == digest
    contenttype = response.getheader('Content-Type')
    return self._parse_response(parser, contenttype, body)