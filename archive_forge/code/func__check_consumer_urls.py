import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_consumer_urls(self, session, sp_response_consumer_url, idp_sp_response_consumer_url):
    """Check if consumer URLs issued by SP and IdP are equal.

        In the initial SAML2 authn Request issued by a Service Provider
        there is a url called ``consumer url``. A trusted Identity Provider
        should issue identical url. If the URLs are not equal the federated
        authn process should be interrupted and the user should be warned.

        :param session: session object to send out HTTP requests.
        :type session: keystoneclient.session.Session
        :param sp_response_consumer_url: consumer URL issued by a SP
        :type  sp_response_consumer_url: string
        :param idp_sp_response_consumer_url: consumer URL issued by an IdP
        :type idp_sp_response_consumer_url: string

        """
    if sp_response_consumer_url != idp_sp_response_consumer_url:
        session.post(sp_response_consumer_url, data=self.SOAP_FAULT, headers=self.ECP_SP_SAML2_REQUEST_HEADERS, authenticated=False)
        msg = _('Consumer URLs from Service Provider %(service_provider)s %(sp_consumer_url)s and Identity Provider %(identity_provider)s %(idp_consumer_url)s are not equal')
        msg = msg % {'service_provider': self.token_url, 'sp_consumer_url': sp_response_consumer_url, 'identity_provider': self.identity_provider, 'idp_consumer_url': idp_sp_response_consumer_url}
        raise exceptions.ValidationError(msg)