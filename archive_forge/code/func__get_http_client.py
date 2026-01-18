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
def _get_http_client(auth_method, session, introspect_endpoint, audience, client_id, func_get_config_option, logger):
    """Get an auth HTTP Client to access the OAuth2.0 Server."""
    if auth_method in _ALL_AUTH_CLIENTS:
        return _ALL_AUTH_CLIENTS.get(auth_method)(session, introspect_endpoint, audience, client_id, func_get_config_option, logger)
    logger.critical('The value is incorrect for option auth_method in group [%s]' % _EXT_AUTH_CONFIG_GROUP_NAME)
    raise ConfigurationError(_('The configuration parameter for key "auth_method" in group [%s] is incorrect.') % _EXT_AUTH_CONFIG_GROUP_NAME)