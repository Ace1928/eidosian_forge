import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
def _setup_verify(self):
    self.verify = libcloud.security.VERIFY_SSL_CERT