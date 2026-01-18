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
def _convert_into_with_meta(self, item, resp):
    if item:
        if isinstance(item, dict):
            return _DictWithMeta(item, resp)
        elif isinstance(item, str):
            return _StrWithMeta(item, resp)
    else:
        return _TupleWithMeta((), resp)