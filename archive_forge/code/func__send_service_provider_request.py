import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _send_service_provider_request(self, session):
    """Initial HTTP GET request to the SAML2 protected endpoint.

        It's crucial to include HTTP headers indicating that the client is
        willing to take advantage of the ECP SAML2 extension and receive data
        as the SOAP.
        Unlike standard authentication methods in the OpenStack Identity,
        the client accesses::
        ``/v3/OS-FEDERATION/identity_providers/{identity_providers}/
        protocols/{protocol}/auth``

        After a successful HTTP call the HTTP response should include SAML2
        authn request in the XML format.

        If a HTTP response contains ``X-Subject-Token`` in the headers and
        the response body is a valid JSON assume the user was already
        authenticated and Keystone returned a valid unscoped token.
        Return True indicating the user was already authenticated.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        """
    sp_response = session.get(self.token_url, headers=self.ECP_SP_EMPTY_REQUEST_HEADERS, authenticated=False)
    if 'X-Subject-Token' in sp_response.headers:
        self.authenticated_response = sp_response
        return True
    try:
        self.saml2_authn_request = etree.XML(sp_response.content)
    except etree.XMLSyntaxError as e:
        msg = _('SAML2: Error parsing XML returned from Service Provider, reason: %s') % e
        raise exceptions.AuthorizationFailure(msg)
    relay_state = self.saml2_authn_request.xpath(self.ECP_RELAY_STATE, namespaces=self.ECP_SAML2_NAMESPACES)
    self.relay_state = self._first(relay_state)
    sp_response_consumer_url = self.saml2_authn_request.xpath(self.ECP_SERVICE_PROVIDER_CONSUMER_URL, namespaces=self.ECP_SAML2_NAMESPACES)
    self.sp_response_consumer_url = self._first(sp_response_consumer_url)
    return False