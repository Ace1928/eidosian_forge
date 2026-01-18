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
def _ensure_called_from_thread(self, queue):
    tid = threading.get_ident()
    for item in queue:
        assert item[0] == tid