from __future__ import (absolute_import, division, print_function)
import os.path
import socket
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from base64 import b64encode
from netrc import netrc
from os import environ
from time import sleep
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import urllib_error
from stat import S_IRUSR, S_IWUSR
from tempfile import gettempdir, NamedTemporaryFile
import yaml
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
def _fetch_conjur_token(conjur_url, account, username, api_key, validate_certs, cert_file):
    conjur_url = '{0}/authn/{1}/{2}/authenticate'.format(conjur_url, account, _encode_str(username))
    display.vvvv('Authentication request to Conjur at: {0}, with user: {1}'.format(conjur_url, _encode_str(username)))
    response = open_url(conjur_url, data=api_key, method='POST', validate_certs=validate_certs, ca_path=cert_file)
    code = response.getcode()
    if code != 200:
        raise AnsibleError("Failed to authenticate as '{0}' (got {1} response)".format(username, code))
    return response.read()