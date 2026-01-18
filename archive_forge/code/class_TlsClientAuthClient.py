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
class TlsClientAuthClient(AbstractAuthClient):
    """Http client with the auth method 'tls_client_auth'."""

    def introspect(self, access_token):
        """Access the introspect API.

        Access the Introspect API to verify the access token by
        the auth method 'tls_client_auth'.
        """
        req_data = {'client_id': self.client_id, 'token': access_token, 'token_type_hint': 'access_token'}
        http_response = self.session.request(self.introspect_endpoint, 'POST', authenticated=False, data=req_data)
        return http_response