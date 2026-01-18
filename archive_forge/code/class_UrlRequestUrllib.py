import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
class UrlRequestUrllib(UrlRequestBase):

    def get_chunks(self, resp, chunk_size, total_size, report_progress, q, trigger, fd=None):
        bytes_so_far = 0
        result = b''
        while 1:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            if fd:
                fd.write(chunk)
            else:
                result += chunk
            bytes_so_far += len(chunk)
            if report_progress:
                q(('progress', resp, (bytes_so_far, total_size)))
                trigger()
            if self._cancel_event.is_set():
                break
        return (bytes_so_far, result)

    def get_response(self, resp):
        return resp.read()

    def get_total_size(self, resp):
        try:
            return int(resp.getheader('content-length'))
        except Exception:
            return -1

    def get_content_type(self, resp):
        return resp.getheader('Content-Type', None)

    def get_status_code(self, resp):
        return resp.status

    def get_all_headers(self, resp):
        return resp.getheaders()

    def close_connection(self, req):
        req.close()

    def _parse_url(self, url):
        parse = urlparse(url)
        host = parse.hostname
        port = parse.port
        userpass = None
        if parse.username and parse.password:
            userpass = {'Authorization': 'Basic {}'.format(b64encode('{}:{}'.format(parse.username, parse.password).encode('utf-8')).decode('utf-8'))}
        return (host, port, userpass, parse)

    def _get_connection_for_scheme(self, scheme):
        """Return the Connection class for a particular scheme.
        This is an internal function that can be expanded to support custom
        schemes.

        Actual supported schemes: http, https.
        """
        if scheme == 'http':
            return HTTPConnection
        elif scheme == 'https' and HTTPSConnection is not None:
            return HTTPSConnection
        else:
            raise Exception('No class for scheme %s' % scheme)

    def call_request(self, body, headers):
        timeout = self._timeout
        ca_file = self.ca_file
        verify = self.verify
        url = self._requested_url
        host, port, userpass, parse = self._parse_url(url)
        if userpass and (not headers):
            headers = userpass
        elif userpass and headers:
            key = list(userpass.keys())[0]
            headers[key] = userpass[key]
        cls = self._get_connection_for_scheme(parse.scheme)
        path = parse.path
        if parse.params:
            path += ';' + parse.params
        if parse.query:
            path += '?' + parse.query
        if parse.fragment:
            path += '#' + parse.fragment
        args = {}
        if timeout is not None:
            args['timeout'] = timeout
        if ca_file is not None and hasattr(ssl, 'create_default_context') and (parse.scheme == 'https'):
            ctx = ssl.create_default_context(cafile=ca_file)
            ctx.verify_mode = ssl.CERT_REQUIRED
            args['context'] = ctx
        if not verify and parse.scheme == 'https' and hasattr(ssl, 'create_default_context'):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            args['context'] = ctx
        if self._proxy_host:
            Logger.debug('UrlRequest: {0} - proxy via {1}:{2}'.format(id(self), self._proxy_host, self._proxy_port))
            req = cls(self._proxy_host, self._proxy_port, **args)
            if parse.scheme == 'https':
                req.set_tunnel(host, port, self._proxy_headers)
            else:
                path = urlunparse(parse)
        else:
            req = cls(host, port, **args)
        method = self._method
        if method is None:
            method = 'GET' if body is None else 'POST'
        req.request(method, path, body, headers or {})
        return (req, req.getresponse())