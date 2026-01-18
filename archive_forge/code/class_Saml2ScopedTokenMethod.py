import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class Saml2ScopedTokenMethod(v3.TokenMethod):
    _method_name = 'saml2'

    def get_auth_data(self, session, auth, headers, **kwargs):
        """Build and return request body for token scoping step."""
        t = super(Saml2ScopedTokenMethod, self).get_auth_data(session, auth, headers, **kwargs)
        _token_method, token = t
        return (self._method_name, token)