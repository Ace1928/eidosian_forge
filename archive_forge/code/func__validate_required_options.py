import argparse
import csv
import glob
from importlib import util as importlib_util
import itertools
import logging
import os
import pkgutil
import sys
from oslo_utils import importutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions as exc
import manilaclient.extension
from manilaclient.v2 import shell as shell_v2
def _validate_required_options(self, tenant_name, tenant_id, project_name, project_id, token, service_catalog_url, auth_url):
    if token and (not service_catalog_url):
        raise exc.CommandError('bypass_url missing: When specifying a token the bypass_url must be set via --bypass-url or env[OS_MANILA_BYPASS_URL]')
    if service_catalog_url and (not token):
        raise exc.CommandError('Token missing: When specifying a bypass_url a token must be set via --os-token or env[OS_TOKEN]')
    if token and service_catalog_url:
        return
    if not (tenant_name or tenant_id or project_name or project_id):
        raise exc.CommandError('You must provide a tenant_name, tenant_id, project_id or project_name (with project_domain_name or project_domain_id) via --os-tenant-name or env[OS_TENANT_NAME], --os-tenant-id or env[OS_TENANT_ID], --os-project-id or env[OS_PROJECT_ID], --os-project-name or env[OS_PROJECT_NAME], --os-project-domain-id or env[OS_PROJECT_DOMAIN_ID] and --os-project-domain-name or env[OS_PROJECT_DOMAIN_NAME].')
    if not auth_url:
        raise exc.CommandError('You must provide an auth url via either --os-auth-url or env[OS_AUTH_URL]')