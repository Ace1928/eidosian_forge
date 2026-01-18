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
class VMDKUploader(Thread):

    def __init__(self, vmdk, url, validate_certs=True, tarinfo=None, create=False):
        Thread.__init__(self)
        self.vmdk = vmdk
        if tarinfo:
            self.size = tarinfo.size
        else:
            self.size = os.stat(vmdk).st_size
        self.url = url
        self.validate_certs = validate_certs
        self.tarinfo = tarinfo
        self.f = None
        self.e = None
        self._create = create

    @property
    def bytes_read(self):
        try:
            return self.f.bytes_read
        except AttributeError:
            return 0

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

    def _open_url(self):
        open_url(self.url, data=self.f, validate_certs=self.validate_certs, **self._request_opts())

    def run(self):
        if self.tarinfo:
            try:
                with TarFileProgressReader(self.vmdk, self.tarinfo) as self.f:
                    self._open_url()
            except Exception:
                self.e = sys.exc_info()
        else:
            try:
                with ProgressReader(self.vmdk, 'rb') as self.f:
                    self._open_url()
            except Exception:
                self.e = sys.exc_info()