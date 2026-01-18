import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _send_idp_saml2_authn_request(self, session):
    """Present modified SAML2 authn assertion from the Service Provider."""
    self._prepare_idp_saml2_request(self.saml2_authn_request)
    idp_saml2_authn_request = self.saml2_authn_request
    idp_response = session.post(self.identity_provider_url, headers={'Content-type': 'text/xml'}, data=etree.tostring(idp_saml2_authn_request), requests_auth=(self.username, self.password), authenticated=False, log=False)
    try:
        self.saml2_idp_authn_response = etree.XML(idp_response.content)
    except etree.XMLSyntaxError as e:
        msg = _('SAML2: Error parsing XML returned from Identity Provider, reason: %s') % e
        raise exceptions.AuthorizationFailure(msg)
    idp_response_consumer_url = self.saml2_idp_authn_response.xpath(self.ECP_IDP_CONSUMER_URL, namespaces=self.ECP_SAML2_NAMESPACES)
    self.idp_response_consumer_url = self._first(idp_response_consumer_url)
    self._check_consumer_urls(session, self.idp_response_consumer_url, self.sp_response_consumer_url)