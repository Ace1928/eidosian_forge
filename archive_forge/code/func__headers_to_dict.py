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
def _headers_to_dict(self, r):
    result = {}
    if hasattr(r, 'getheaders'):
        for n, v in r.getheaders():
            result[n.lower()] = v.strip()
    else:
        for line in r.info().headers:
            if line.find(':') != -1:
                n, v = line.split(': ', 1)
                result[n.lower()] = v.strip()
    return result