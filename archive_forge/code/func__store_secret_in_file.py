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
def _store_secret_in_file(value):
    secrets_file = NamedTemporaryFile(mode='w', dir=_default_tmp_path(), delete=False)
    os.chmod(secrets_file.name, S_IRUSR | S_IWUSR)
    secrets_file.write(value[0])
    return [secrets_file.name]