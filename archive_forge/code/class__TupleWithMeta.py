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
class _TupleWithMeta(tuple, _RequestIdMixin):

    def __new__(cls, values, resp):
        return super(_TupleWithMeta, cls).__new__(cls, values)

    def __init__(self, values, resp):
        self._request_ids_setup()
        self._append_request_ids(resp)