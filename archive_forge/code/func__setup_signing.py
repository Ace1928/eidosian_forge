import os
import warnings
import requests
from requests.adapters import HTTPAdapter
import libcloud.security
from libcloud.utils.py3 import urlparse
def _setup_signing(self, cert_file=None, key_file=None):
    """
        Setup request signing by mounting a signing
        adapter to the session
        """
    self.session.mount('https://', SignedHTTPSAdapter(cert_file, key_file))