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
def _request_opts(self):
    """
        Requests for vmdk files differ from other file types. Build the request options here to handle that
        """
    headers = {'Content-Length': self.size, 'Content-Type': 'application/octet-stream'}
    if self._create:
        method = 'PUT'
        headers['Overwrite'] = 't'
    else:
        method = 'POST'
        headers['Content-Type'] = 'application/x-vnd.vmware-streamVmdk'
    return {'method': method, 'headers': headers}