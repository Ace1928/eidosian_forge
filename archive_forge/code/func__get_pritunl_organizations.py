from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def _get_pritunl_organizations(api_token, api_secret, base_url, validate_certs=True):
    return pritunl_auth_request(base_url=base_url, api_token=api_token, api_secret=api_secret, method='GET', path='/organization', validate_certs=validate_certs)