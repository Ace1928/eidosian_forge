import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
class _Reader(object):

    def __init__(self, data, hashing_algo, verifier=None):
        self._size = 0
        self.data = data
        self.os_hash_value = utils.get_hasher(hashing_algo, False)
        self.checksum = utils.get_hasher('md5', False)
        self.verifier = verifier

    def read(self, size=None):
        result = self.data.read(size)
        self._size += len(result)
        self.checksum.update(result)
        self.os_hash_value.update(result)
        if self.verifier:
            self.verifier.update(result)
        return result

    @property
    def size(self):
        return self._size