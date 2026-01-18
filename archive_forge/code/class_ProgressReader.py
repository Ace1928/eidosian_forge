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
class ProgressReader(io.FileIO):

    def __init__(self, name, mode='r', closefd=True):
        self.bytes_read = 0
        io.FileIO.__init__(self, name, mode=mode, closefd=closefd)

    def read(self, size=10240):
        chunk = io.FileIO.read(self, size)
        self.bytes_read += len(chunk)
        return chunk