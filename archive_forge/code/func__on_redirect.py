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
def _on_redirect(self, req, *args):
    self.queue.append((threading.get_ident(), 'redirect', args))