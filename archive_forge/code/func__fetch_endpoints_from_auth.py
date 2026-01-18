import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def _fetch_endpoints_from_auth(self, url):
    """We have a token, but don't know the final endpoint for
        the region. We have to go back to the auth service and
        ask again. This request requires an admin-level token
        to work. The proxy token supplied could be from a low-level enduser.

        We can't get this from the keystone service endpoint, we have to use
        the admin endpoint.

        This will overwrite our admin token with the user token.
        """
    url = '/'.join([url, 'tokens', '%s?belongsTo=%s' % (self.proxy_token, self.proxy_tenant_id)])
    self._logger.debug('Using Endpoint URL: %s' % url)
    resp, body = self.request(url, 'GET', headers={'X-Auth-Token': self.auth_token})
    return self._extract_service_catalog(url, resp, body, extract_token=False)