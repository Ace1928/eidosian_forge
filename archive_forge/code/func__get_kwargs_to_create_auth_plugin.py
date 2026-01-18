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
def _get_kwargs_to_create_auth_plugin(self, args):
    if not args.os_username:
        raise exc.CommandError(_('You must provide a username via either --os-username or env[OS_USERNAME]'))
    if not args.os_password:
        if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
            try:
                args.os_password = getpass.getpass('OS Password: ')
            except EOFError:
                pass
        if not args.os_password:
            raise exc.CommandError(_('You must provide a password via either --os-password, env[OS_PASSWORD], or prompted response'))
    os_project_name = getattr(args, 'os_project_name', getattr(args, 'os_tenant_name', None))
    os_project_id = getattr(args, 'os_project_id', getattr(args, 'os_tenant_id', None))
    if not any([os_project_name, os_project_id]):
        raise exc.CommandError(_('You must provide a project_id or project_name (with project_domain_name or project_domain_id) via   --os-project-id (env[OS_PROJECT_ID])  --os-project-name (env[OS_PROJECT_NAME]),  --os-project-domain-id (env[OS_PROJECT_DOMAIN_ID])  --os-project-domain-name (env[OS_PROJECT_DOMAIN_NAME])'))
    if not args.os_auth_url:
        raise exc.CommandError(_('You must provide an auth url via either --os-auth-url or via env[OS_AUTH_URL]'))
    kwargs = {'auth_url': args.os_auth_url, 'username': args.os_username, 'user_id': args.os_user_id, 'user_domain_id': args.os_user_domain_id, 'user_domain_name': args.os_user_domain_name, 'password': args.os_password, 'tenant_name': args.os_tenant_name, 'tenant_id': args.os_tenant_id, 'project_name': args.os_project_name, 'project_id': args.os_project_id, 'project_domain_name': args.os_project_domain_name, 'project_domain_id': args.os_project_domain_id}
    return kwargs