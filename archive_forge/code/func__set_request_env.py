import abc
import copy
import hashlib
import os
import ssl
import time
import uuid
import jwt.utils
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests.auth
import webob.dec
import webob.exc
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.exceptions import ConfigurationError
from keystonemiddleware.exceptions import KeystoneMiddlewareException
from keystonemiddleware.i18n import _
def _set_request_env(self, request, token_data):
    """Set request.environ with the necessary information."""
    request.environ['external.token_info'] = token_data
    request.environ['HTTP_X_IDENTITY_STATUS'] = 'Confirmed'
    request.environ['HTTP_X_ROLES'] = token_data.get('roles')
    request.environ['HTTP_X_ROLE'] = token_data.get('roles')
    request.environ['HTTP_X_USER_ID'] = token_data.get('user_id')
    request.environ['HTTP_X_USER_NAME'] = token_data.get('user_name')
    request.environ['HTTP_X_USER_DOMAIN_ID'] = token_data.get('user_domain_id')
    request.environ['HTTP_X_USER_DOMAIN_NAME'] = token_data.get('user_domain_name')
    if token_data.get('is_admin') == 'true':
        request.environ['HTTP_X_IS_ADMIN_PROJECT'] = token_data.get('is_admin')
    request.environ['HTTP_X_USER'] = token_data.get('user_name')
    if token_data.get('system_scope'):
        request.environ['HTTP_OPENSTACK_SYSTEM_SCOPE'] = token_data.get('system_scope')
    elif token_data.get('project_id'):
        request.environ['HTTP_X_PROJECT_ID'] = token_data.get('project_id')
        request.environ['HTTP_X_PROJECT_NAME'] = token_data.get('project_name')
        request.environ['HTTP_X_PROJECT_DOMAIN_ID'] = token_data.get('project_domain_id')
        request.environ['HTTP_X_PROJECT_DOMAIN_NAME'] = token_data.get('project_domain_name')
        request.environ['HTTP_X_TENANT_ID'] = token_data.get('project_id')
        request.environ['HTTP_X_TENANT_NAME'] = token_data.get('project_name')
        request.environ['HTTP_X_TENANT'] = token_data.get('project_id')
    else:
        request.environ['HTTP_X_DOMAIN_ID'] = token_data.get('domain_id')
        request.environ['HTTP_X_DOMAIN_NAME'] = token_data.get('domain_name')
    self._log.debug('The access token data is %s.' % jsonutils.dumps(token_data))