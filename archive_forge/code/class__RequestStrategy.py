import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from keystoneclient.v3 import client as v3_client
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.i18n import _
class _RequestStrategy(object):
    AUTH_VERSION = None

    def __init__(self, adap, include_service_catalog=None, requested_auth_interface=None):
        self._include_service_catalog = include_service_catalog
        self._requested_auth_interface = requested_auth_interface

    def verify_token(self, user_token, allow_expired=False):
        pass