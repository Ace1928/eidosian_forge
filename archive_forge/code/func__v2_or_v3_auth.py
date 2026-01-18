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
def _v2_or_v3_auth(self, url):
    """Authenticate against a v2.0 auth service."""
    if self.ks_version == 'v3':
        body = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'domain': {'name': self.user_domain_name}, 'name': self.user, 'password': self.password}}}}}
        scope = {'project': {'domain': {'name': self.project_domain_name}}}
        if self.projectid:
            scope['project']['name'] = self.projectid
        elif self.tenant_id:
            scope['project']['id'] = self.tenant_id
        body['auth']['scope'] = scope
    else:
        body = {'auth': {'passwordCredentials': {'username': self.user, 'password': self.password}}}
        if self.projectid:
            body['auth']['tenantName'] = self.projectid
        elif self.tenant_id:
            body['auth']['tenantId'] = self.tenant_id
    return self._authenticate(url, body)