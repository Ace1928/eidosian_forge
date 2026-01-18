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
class UrlRequestRequests(UrlRequestBase):

    def get_chunks(self, resp, chunk_size, total_size, report_progress, q, trigger, fd=None):
        bytes_so_far = 0
        result = b''
        for chunk in resp.iter_content(chunk_size):
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
        return resp.content

    def get_total_size(self, resp):
        return int(resp.headers.get('Content-Length', -1))

    def get_content_type(self, resp):
        return resp.headers.get('Content-Type', None)

    def get_status_code(self, resp):
        return resp.status_code

    def get_all_headers(self, resp):
        return resp.headers.items()

    def close_connection(self, req):
        pass

    def call_request(self, body, headers):
        timeout = self._timeout
        ca_file = self.ca_file
        verify = self.verify
        url = self._requested_url
        auth = self._auth
        req = requests
        kwargs = {}
        if self._method is None:
            method = 'get' if body is None else 'post'
        else:
            method = self._method.lower()
        req_call = getattr(req, method)
        if auth:
            kwargs['auth'] = auth
        response = req_call(url, data=body, headers=headers, timeout=timeout, verify=verify, cert=ca_file, **kwargs)
        return (None, response)