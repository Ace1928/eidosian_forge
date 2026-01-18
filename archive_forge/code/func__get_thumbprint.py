from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _get_thumbprint(self, hostname):
    pem = ssl.get_server_certificate((hostname, 443))
    sha1 = hashlib.sha1(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()
    colon_notion = ':'.join((sha1[i:i + 2] for i in range(0, len(sha1), 2)))
    return None if sha1 is None else colon_notion