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
def _get_keystone_auth_plugin(self, ks_session, **kwargs):
    auth_url = kwargs.pop('auth_url', None)
    v2_auth_url, v3_auth_url = self._discover_auth_versions(session=ks_session, auth_url=auth_url)
    user_id = kwargs.pop('user_id', None)
    username = kwargs.pop('username', None)
    password = kwargs.pop('password', None)
    user_domain_name = kwargs.pop('user_domain_name', None)
    user_domain_id = kwargs.pop('user_domain_id', None)
    project_id = kwargs.pop('project_id', None) or kwargs.pop('tenant_id', None)
    project_name = kwargs.pop('project_name', None) or kwargs.pop('tenant_name', None)
    project_domain_id = kwargs.pop('project_domain_id', None)
    project_domain_name = kwargs.pop('project_domain_name', None)
    auth = None
    use_domain = user_domain_id or user_domain_name or project_domain_id or project_domain_name
    use_v3 = v3_auth_url and (use_domain or not v2_auth_url)
    use_v2 = v2_auth_url and (not use_domain)
    if use_v3:
        auth = v3_auth.Password(v3_auth_url, user_id=user_id, username=username, password=password, user_domain_id=user_domain_id, user_domain_name=user_domain_name, project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
    elif use_v2:
        auth = v2_auth.Password(v2_auth_url, username, password, tenant_id=project_id, tenant_name=project_name)
    else:
        exc.CommandError('Credential and auth_url mismatch. The given auth_url is using Keystone V2 endpoint, which may not able to handle Keystone V3 credentials. Please provide a correct Keystone V3 auth_url.')
    return auth