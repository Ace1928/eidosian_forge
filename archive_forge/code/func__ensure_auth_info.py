import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def _ensure_auth_info(self, args):
    if not cliutils.isunauthenticated(args.func):
        if not (args.os_token and (args.os_auth_url or args.os_endpoint_override)) and (not args.os_cloud):
            if not (args.os_username or args.os_user_id):
                raise exc.CommandError('You must provide a username via either --os-username or via env[OS_USERNAME]')
            if not args.os_password:
                raise exc.CommandError('You must provide a password via either --os-password, env[OS_PASSWORD], or prompted response')
            if not args.os_project_name and (not args.os_project_id):
                raise exc.CommandError('You must provide a project name or project id via --os-project-name, --os-project-id, env[OS_PROJECT_NAME] or env[OS_PROJECT_ID]')
            if not args.os_auth_url:
                raise exc.CommandError('You must provide an auth url via either --os-auth-url or via env[OS_AUTH_URL]')