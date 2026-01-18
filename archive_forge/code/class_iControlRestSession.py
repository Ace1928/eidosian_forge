from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
class iControlRestSession(object):
    """Represents a session that communicates with a BigIP.

    This acts as a loose wrapper around Ansible's ``Request`` class. We're doing
    this as interim work until we move to the httpapi connector.
    """

    def __init__(self, headers=None, use_proxy=True, force=False, timeout=120, validate_certs=True, url_username=None, url_password=None, http_agent=None, force_basic_auth=False, follow_redirects='urllib2', client_cert=None, client_key=None, cookies=None):
        self.request = Request(headers=headers, use_proxy=use_proxy, force=force, timeout=timeout, validate_certs=validate_certs, url_username=url_username, url_password=url_password, http_agent=http_agent, force_basic_auth=force_basic_auth, follow_redirects=follow_redirects, client_cert=client_cert, client_key=client_key, cookies=cookies)
        self.last_url = None

    def get_headers(self, result):
        try:
            return dict(result.getheaders())
        except AttributeError:
            return result.headers

    def update_response(self, response, result):
        response.headers = self.get_headers(result)
        response._content = result.read()
        response.status = result.getcode()
        response.url = result.geturl()
        response.msg = 'OK (%s bytes)' % response.headers.get('Content-Length', 'unknown')

    def send(self, method, url, **kwargs):
        response = Response()
        self.last_url = url
        body = None
        data = kwargs.pop('data', None)
        json = kwargs.pop('json', None)
        if not data and json is not None:
            self.request.headers.update(BASE_HEADERS)
            body = _json.dumps(json)
            if not isinstance(body, bytes):
                body = body.encode('utf-8')
        if data:
            body = data
        if body:
            kwargs['data'] = body
        try:
            result = self.request.open(method, url, **kwargs)
        except HTTPError as e:
            self.update_response(response, e)
            return response
        self.update_response(response, result)
        return response

    def delete(self, url, **kwargs):
        return self.send('DELETE', url, **kwargs)

    def get(self, url, **kwargs):
        return self.send('GET', url, **kwargs)

    def patch(self, url, data=None, **kwargs):
        return self.send('PATCH', url, data=data, **kwargs)

    def post(self, url, data=None, **kwargs):
        return self.send('POST', url, data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        return self.send('PUT', url, data=data, **kwargs)

    def __del__(self):
        if self.last_url is None:
            return
        token = self.request.headers.get('X-F5-Auth-Token', None)
        if not token:
            return
        try:
            p = generic_urlparse(urlparse(self.last_url))
            uri = 'https://{0}:{1}{2}{3}'.format(p['hostname'], p['port'], LOGOUT, token)
            self.delete(uri)
        except ValueError:
            pass