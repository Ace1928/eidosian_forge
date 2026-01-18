from __future__ import absolute_import, division, print_function
import base64
import hashlib
import hmac
import json
import time
import uuid
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import open_url
def _post_pritunl_user(api_token, api_secret, base_url, organization_id, user_data, validate_certs=True):
    return pritunl_auth_request(api_token=api_token, api_secret=api_secret, base_url=base_url, method='POST', path='/user/%s' % organization_id, headers={'Content-Type': 'application/json'}, data=json.dumps(user_data), validate_certs=validate_certs)