import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class Saml2UnscopedTokenAuthMethod(v3.AuthMethod):
    _method_parameters = []

    def get_auth_data(self, session, auth, headers, **kwargs):
        raise exceptions.MethodNotImplemented(_('This method should never be called'))