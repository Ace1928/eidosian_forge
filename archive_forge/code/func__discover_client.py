import argparse
import collections
import getpass
import logging
import sys
from urllib import parse as urlparse
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
from keystoneauth1 import session
from oslo_utils import importutils
import requests
import cinderclient
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import client
from cinderclient import exceptions as exc
from cinderclient import utils
def _discover_client(self, current_client, os_api_version, os_endpoint_type, os_service_type, os_username, os_password, os_project_name, os_auth_url, client_args):
    discovered_version = api_versions.discover_version(current_client, os_api_version)
    if not os_endpoint_type:
        os_endpoint_type = DEFAULT_CINDER_ENDPOINT_TYPE
    if not os_service_type:
        os_service_type = self._discover_service_type(discovered_version)
    API_MAX_VERSION = api_versions.APIVersion(api_versions.MAX_VERSION)
    if discovered_version != API_MAX_VERSION or os_service_type != 'volume' or os_endpoint_type != DEFAULT_CINDER_ENDPOINT_TYPE:
        client_args['service_type'] = os_service_type
        client_args['endpoint_type'] = os_endpoint_type
        return (client.Client(discovered_version, os_username, os_password, os_project_name, os_auth_url, **client_args), discovered_version)
    else:
        return (current_client, discovered_version)