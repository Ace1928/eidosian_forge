import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import certifi
import pytest
import responses
from kivy.network.urlrequest import UrlRequestRequests as UrlRequest
from requests.auth import HTTPBasicAuth
from responses import matchers
class UrlRequestQueue:

    def __init__(self, queue):
        self.queue = queue

    def _on_success(self, req, *args):
        self.queue.append((threading.get_ident(), 'success', args))

    def _on_redirect(self, req, *args):
        self.queue.append((threading.get_ident(), 'redirect', args))

    def _on_error(self, req, *args):
        self.queue.append((threading.get_ident(), 'error', args))

    def _on_failure(self, req, *args):
        self.queue.append((threading.get_ident(), 'failure', args))

    def _on_progress(self, req, *args):
        self.queue.append((threading.get_ident(), 'progress', args))

    def _on_finish(self, req, *args):
        self.queue.append((threading.get_ident(), 'finish', args))