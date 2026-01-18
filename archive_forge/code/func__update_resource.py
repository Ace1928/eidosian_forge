import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def _update_resource(self, path, **kwargs):
    revision_number = kwargs.pop('revision_number', None)
    if revision_number:
        headers = kwargs.setdefault('headers', {})
        headers['If-Match'] = 'revision_number=%s' % revision_number
    return self.put(path, **kwargs)