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
def _get_env_credentials(self):
    env_credentials = dict()
    for attribute, env_variable in AZURE_CREDENTIAL_ENV_MAPPING.items():
        env_credentials[attribute] = os.environ.get(env_variable, None)
    if env_credentials['profile']:
        credentials = self._get_profile(env_credentials['profile'])
        return credentials
    if env_credentials.get('subscription_id') is not None:
        return env_credentials
    return None