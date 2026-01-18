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
def _load_identity_from_file(identity_path, appliance_url):
    display.vvvv('identity file: {0}'.format(identity_path))
    if not os.path.exists(identity_path):
        return {}
    display.vvvv('Loading identity from: {0} for {1}'.format(identity_path, appliance_url))
    conjur_authn_url = '{0}/authn'.format(appliance_url)
    identity = netrc(identity_path)
    if identity.authenticators(conjur_authn_url) is None:
        raise AnsibleError('The netrc file on the controlling host does not contain an entry for: {0}'.format(conjur_authn_url))
    id, account, api_key = identity.authenticators(conjur_authn_url)
    if not id or not api_key:
        return {}
    return {'id': id, 'api_key': api_key}