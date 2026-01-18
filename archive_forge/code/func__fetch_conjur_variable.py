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
def _fetch_conjur_variable(conjur_variable, token, conjur_url, account, validate_certs, cert_file):
    token = b64encode(token)
    headers = {'Authorization': 'Token token="{0}"'.format(token.decode('utf-8'))}
    url = '{0}/secrets/{1}/variable/{2}'.format(conjur_url, account, _encode_str(conjur_variable))
    display.vvvv('Conjur Variable URL: {0}'.format(url))
    response = _repeat_open_url(url, headers=headers, method='GET', validate_certs=validate_certs, ca_path=cert_file)
    if response.getcode() == 200:
        display.vvvv('Conjur variable {0} was successfully retrieved'.format(conjur_variable))
        value = response.read().decode('utf-8')
        return [value]
    if response.getcode() == 401:
        raise AnsibleError('Conjur request has invalid authorization credentials')
    if response.getcode() == 403:
        raise AnsibleError("The controlling host's Conjur identity does not have authorization to retrieve {0}".format(conjur_variable))
    if response.getcode() == 404:
        raise AnsibleError('The variable {0} does not exist'.format(conjur_variable))
    return {}