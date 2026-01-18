from distutils.version import StrictVersion
import functools
from http import client as http_client
import json
import logging
import re
import textwrap
import time
from urllib import parse as urlparse
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common.i18n import _
from ironicclient import exc
def _query_server(conn):
    if self.os_ironic_api_version and (not isinstance(self.os_ironic_api_version, list)) and (self.os_ironic_api_version != 'latest'):
        base_version = '/v%s' % str(self.os_ironic_api_version).split('.')[0]
    else:
        base_version = API_VERSION
    resp = self._make_simple_request(conn, 'GET', base_version)
    if not resp.ok:
        raise exc.from_response(resp, method='GET', url=base_version)
    return resp