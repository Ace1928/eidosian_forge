import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class Saml2ScopedToken(v3.Token):
    """Class for scoping unscoped saml2 token."""
    _auth_method_class = Saml2ScopedTokenMethod

    def __init__(self, auth_url, token, **kwargs):
        super(Saml2ScopedToken, self).__init__(auth_url, token, **kwargs)
        if not (self.project_id or self.domain_id):
            raise exceptions.ValidationError(_('Neither project nor domain specified'))