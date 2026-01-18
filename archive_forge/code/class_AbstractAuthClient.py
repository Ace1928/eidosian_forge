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
class AbstractAuthClient(object, metaclass=abc.ABCMeta):
    """Abstract http client using to access the OAuth2.0 Server."""

    def __init__(self, session, introspect_endpoint, audience, client_id, func_get_config_option, logger):
        self.session = session
        self.introspect_endpoint = introspect_endpoint
        self.audience = audience
        self.client_id = client_id
        self.get_config_option = func_get_config_option
        self.logger = logger

    @abc.abstractmethod
    def introspect(self, access_token):
        """Access the introspect API."""
        pass