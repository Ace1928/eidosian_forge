from the request environment and it's identified by the ``swift.cache`` key.
import copy
import re
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.auth_token import _identity
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.auth_token import _request
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.i18n import _
def _do_fetch_token(self, token, **kwargs):
    """Helper method to fetch a token and convert it into an AccessInfo."""
    token = token.strip()
    data = self.fetch_token(token, **kwargs)
    try:
        return (data, access.create(body=data, auth_token=token))
    except Exception:
        self.log.warning('Invalid token contents.', exc_info=True)
        raise ksm_exceptions.InvalidToken(_('Token authorization failed'))