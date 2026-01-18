import argparse
import copy
import getpass
import hashlib
import json
import logging
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import importutils
import urllib.parse
import glanceclient
from glanceclient._i18n import _
from glanceclient.common import utils
from glanceclient import exc
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
def _get_versioned_client(self, api_version, args):
    endpoint = self._get_image_url(args)
    auth_token = args.os_auth_token
    if endpoint and auth_token:
        kwargs = {'token': auth_token, 'insecure': args.insecure, 'timeout': args.timeout, 'cacert': args.os_cacert, 'cert': args.os_cert, 'key': args.os_key}
    else:
        ks_session = loading.load_session_from_argparse_arguments(args)
        auth_plugin_kwargs = self._get_kwargs_to_create_auth_plugin(args)
        ks_session.auth = self._get_keystone_auth_plugin(ks_session=ks_session, **auth_plugin_kwargs)
        kwargs = {'session': ks_session}
        if endpoint is None:
            endpoint_type = args.os_endpoint_type or 'public'
            service_type = args.os_service_type or 'image'
            endpoint = ks_session.get_endpoint(service_type=service_type, interface=endpoint_type, region_name=args.os_region_name)
    return glanceclient.Client(api_version, endpoint, **kwargs)