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
def _fetch_url(self, url, body, headers, q):
    trigger = self._trigger_result
    chunk_size = self._chunk_size
    report_progress = self.on_progress is not None
    file_path = self.file_path
    if self._debug:
        Logger.debug('UrlRequest: {0} Fetch url <{1}>'.format(id(self), url))
        Logger.debug('UrlRequest: {0} - body: {1}'.format(id(self), body))
        Logger.debug('UrlRequest: {0} - headers: {1}'.format(id(self), headers))
    req, resp = self.call_request(body, headers)
    if report_progress or file_path is not None:
        total_size = self.get_total_size(resp)
        if report_progress:
            q(('progress', resp, (0, total_size)))
        if file_path is not None:
            with open(file_path, 'wb') as fd:
                bytes_so_far, result = self.get_chunks(resp, chunk_size, total_size, report_progress, q, trigger, fd=fd)
        else:
            bytes_so_far, result = self.get_chunks(resp, chunk_size, total_size, report_progress, q, trigger)
        if report_progress:
            q(('progress', resp, (bytes_so_far, total_size)))
            trigger()
    else:
        result = self.get_response(resp)
        try:
            if isinstance(result, bytes):
                result = result.decode('utf-8')
        except UnicodeDecodeError:
            pass
    self.close_connection(req)
    return (result, resp)