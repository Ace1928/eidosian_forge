import logging
import os
import debtcollector.renames
from keystoneauth1 import access
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
import requests
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _check_uri_length(self, url):
    uri_len = len(self.endpoint_url) + len(url)
    if uri_len > MAX_URI_LEN:
        raise exceptions.RequestURITooLong(excess=uri_len - MAX_URI_LEN)