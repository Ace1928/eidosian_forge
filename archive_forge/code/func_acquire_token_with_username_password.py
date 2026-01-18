from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def acquire_token_with_username_password(self, authority, username, password, client_id, tenant):
    authority_uri = authority
    if tenant is not None:
        authority_uri = authority + '/' + tenant
    context = ClientApplication(client_id=client_id, authority=authority_uri)
    base_url = self._cloud_environment.endpoints.resource_manager
    if not base_url.endswith('/'):
        base_url += '/'
    scopes = [base_url + '.default']
    token_response = context.acquire_token_by_username_password(username, password, scopes)
    return AADTokenCredentials(token_response)