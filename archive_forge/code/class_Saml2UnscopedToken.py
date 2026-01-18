import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class Saml2UnscopedToken(_BaseSAMLPlugin):
    """Implement authentication plugin for SAML2 protocol.

    ECP stands for `Enhanced Client or Proxy` and is a SAML2 extension
    for federated authentication where a transportation layer consists of
    HTTP protocol and XML SOAP messages.

    `Read for more information
    <https://wiki.shibboleth.net/confluence/display/SHIB2/ECP>`_ on ECP.

    Reference the `SAML2 ECP specification <https://www.oasis-open.org/\\
    committees/download.php/49979/saml-ecp-v2.0-wd09.pdf>`_.

    Currently only HTTPBasicAuth mechanism is available for the IdP
    authenication.

    :param auth_url: URL of the Identity Service
    :type auth_url: string

    :param identity_provider: name of the Identity Provider the client will
                              authenticate against. This parameter will be used
                              to build a dynamic URL used to obtain unscoped
                              OpenStack token.
    :type identity_provider: string

    :param identity_provider_url: An Identity Provider URL, where the SAML2
                                  authn request will be sent.
    :type identity_provider_url: string

    :param username: User's login
    :type username: string

    :param password: User's password
    :type password: string

    """
    _auth_method_class = Saml2UnscopedTokenAuthMethod
    SAML2_HEADER_INDEX = 0
    ECP_SP_EMPTY_REQUEST_HEADERS = {'Accept': 'text/html, application/vnd.paos+xml', 'PAOS': 'ver="urn:liberty:paos:2003-08";"urn:oasis:names:tc:SAML:2.0:profiles:SSO:ecp"'}
    ECP_SP_SAML2_REQUEST_HEADERS = {'Content-Type': 'application/vnd.paos+xml'}
    ECP_SAML2_NAMESPACES = {'ecp': 'urn:oasis:names:tc:SAML:2.0:profiles:SSO:ecp', 'S': 'http://schemas.xmlsoap.org/soap/envelope/', 'paos': 'urn:liberty:paos:2003-08'}
    ECP_RELAY_STATE = '//ecp:RelayState'
    ECP_SERVICE_PROVIDER_CONSUMER_URL = '/S:Envelope/S:Header/paos:Request/@responseConsumerURL'
    ECP_IDP_CONSUMER_URL = '/S:Envelope/S:Header/ecp:Response/@AssertionConsumerServiceURL'
    SOAP_FAULT = '\n    <S:Envelope xmlns:S="http://schemas.xmlsoap.org/soap/envelope/">\n       <S:Body>\n         <S:Fault>\n            <faultcode>S:Server</faultcode>\n            <faultstring>responseConsumerURL from SP and\n            assertionConsumerServiceURL from IdP do not match\n            </faultstring>\n         </S:Fault>\n       </S:Body>\n    </S:Envelope>\n    '

    def __init__(self, auth_url, identity_provider, identity_provider_url, username, password, **kwargs):
        super(Saml2UnscopedToken, self).__init__(auth_url=auth_url, **kwargs)
        self.identity_provider = identity_provider
        self.identity_provider_url = identity_provider_url
        self._username, self._password = (username, password)

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        self._password = value

    def _handle_http_ecp_redirect(self, session, response, method, **kwargs):
        if response.status_code not in (self.HTTP_MOVED_TEMPORARILY, self.HTTP_SEE_OTHER):
            return response
        location = response.headers['location']
        return session.request(location, method, authenticated=False, **kwargs)

    def _prepare_idp_saml2_request(self, saml2_authn_request):
        header = saml2_authn_request[self.SAML2_HEADER_INDEX]
        saml2_authn_request.remove(header)

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

    def _send_service_provider_saml2_authn_response(self, session):
        """Present SAML2 assertion to the Service Provider.

        The assertion is issued by a trusted Identity Provider for the
        authenticated user. This function directs the HTTP request to SP
        managed URL, for instance: ``https://<host>:<port>/Shibboleth.sso/
        SAML2/ECP``.
        Upon success there's a session created and access to the protected
        resource is granted. Many implementations of the SP return HTTP 302/303
        status code pointing to the protected URL (``https://<host>:<port>/v3/
        OS-FEDERATION/identity_providers/{identity_provider}/protocols/
        {protocol_id}/auth`` in this case). Saml2 plugin should point to that
        URL again, with HTTP GET method, expecting an unscoped token.

        :param session: a session object to send out HTTP requests.

        """
        self.saml2_idp_authn_response[0][0] = self.relay_state
        response = session.post(self.idp_response_consumer_url, headers=self.ECP_SP_SAML2_REQUEST_HEADERS, data=etree.tostring(self.saml2_idp_authn_response), authenticated=False, redirect=False)
        response = self._handle_http_ecp_redirect(session, response, method='GET', headers=self.ECP_SP_SAML2_REQUEST_HEADERS)
        self.authenticated_response = response

    def _get_unscoped_token(self, session):
        """Get unscoped OpenStack token after federated authentication.

        This is a multi-step process including multiple HTTP requests.

        The federated authentication consists of::
        * HTTP GET request to the Identity Service (acting as a Service
          Provider). Client utilizes URL::
          ``/v3/OS-FEDERATION/identity_providers/{identity_provider}/
          protocols/saml2/auth``.
          It's crucial to include HTTP headers indicating we are expecting
          SOAP message in return.
          Service Provider should respond with such SOAP message.
          This step is handed by a method
          ``Saml2UnscopedToken_send_service_provider_request()``

        * HTTP POST request to the external Identity Provider service with
          ECP extension enabled. The content sent is a header removed SOAP
          message  returned from the Service Provider. It's also worth noting
          that ECP extension to the SAML2 doesn't define authentication method.
          The most popular is HttpBasicAuth with just user and password.
          Other possibilities could be X509 certificates or Kerberos.
          Upon successful authentication the user should receive a SAML2
          assertion.
          This step is handed by a method
          ``Saml2UnscopedToken_send_idp_saml2_authn_request(session)``

        * HTTP POST request again to the Service Provider. The body of the
          request includes SAML2 assertion issued by a trusted Identity
          Provider. The request should be sent to the Service Provider
          consumer url specified in the SAML2 assertion.
          Providing the authentication was successful and both Service Provider
          and Identity Providers are trusted to each other, the Service
          Provider will issue an unscoped token with a list of groups the
          federated user is a member of.
          This step is handed by a method
          ``Saml2UnscopedToken_send_service_provider_saml2_authn_response()``

          Unscoped token example::

            {
                "token": {
                    "methods": [
                        "saml2"
                    ],
                    "user": {
                        "id": "username%40example.com",
                        "name": "username@example.com",
                        "OS-FEDERATION": {
                            "identity_provider": "ACME",
                            "protocol": "saml2",
                            "groups": [
                                {"id": "abc123"},
                                {"id": "bcd234"}
                            ]
                        }
                    }
                }
            }


        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :returns: (token, token_json)

        """
        saml_authenticated = self._send_service_provider_request(session)
        if not saml_authenticated:
            self._send_idp_saml2_authn_request(session)
            self._send_service_provider_saml2_authn_response(session)
        return (self.authenticated_response.headers['X-Subject-Token'], self.authenticated_response.json()['token'])

    def get_auth_ref(self, session, **kwargs):
        """Authenticate via SAML2 protocol and retrieve unscoped token.

        This is a multi-step process where a client does federated authn
        receives an unscoped token.

        Federated authentication utilizing SAML2 Enhanced Client or Proxy
        extension. See ``Saml2UnscopedToken_get_unscoped_token()``
        for more information on that step.
        Upon successful authentication and assertion mapping an
        unscoped token is returned and stored within the plugin object for
        further use.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :return: an object with scoped token's id and unscoped token json
                 included.
        :rtype: :py:class:`keystoneclient.access.AccessInfoV3`

        """
        token, token_json = self._get_unscoped_token(session)
        return access.AccessInfoV3(token, **token_json)