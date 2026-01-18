import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
def _check_env_value_system_scope(self, request_environ, user_id, user_name, user_domain_id, user_domain_name, roles, is_admin=True, system_scope=True):
    self.assertEqual('Confirmed', request_environ['HTTP_X_IDENTITY_STATUS'])
    self.assertEqual(roles, request_environ['HTTP_X_ROLES'])
    self.assertEqual(roles, request_environ['HTTP_X_ROLE'])
    self.assertEqual(user_id, request_environ['HTTP_X_USER_ID'])
    self.assertEqual(user_name, request_environ['HTTP_X_USER_NAME'])
    self.assertEqual(user_domain_id, request_environ['HTTP_X_USER_DOMAIN_ID'])
    self.assertEqual(user_domain_name, request_environ['HTTP_X_USER_DOMAIN_NAME'])
    if is_admin:
        self.assertEqual('true', request_environ['HTTP_X_IS_ADMIN_PROJECT'])
    else:
        self.assertNotIn('HTTP_X_IS_ADMIN_PROJECT', request_environ)
    self.assertEqual(user_name, request_environ['HTTP_X_USER'])
    self.assertEqual('all', request_environ['HTTP_OPENSTACK_SYSTEM_SCOPE'])
    self.assertNotIn('HTTP_X_DOMAIN_ID', request_environ)
    self.assertNotIn('HTTP_X_DOMAIN_NAME', request_environ)
    self.assertNotIn('HTTP_X_PROJECT_ID', request_environ)
    self.assertNotIn('HTTP_X_PROJECT_NAME', request_environ)
    self.assertNotIn('HTTP_X_PROJECT_DOMAIN_ID', request_environ)
    self.assertNotIn('HTTP_X_PROJECT_DOMAIN_NAME', request_environ)
    self.assertNotIn('HTTP_X_TENANT_ID', request_environ)
    self.assertNotIn('HTTP_X_TENANT_NAME', request_environ)
    self.assertNotIn('HTTP_X_TENANT', request_environ)