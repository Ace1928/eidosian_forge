import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
class ADFSUnscopedToken(_BaseSAMLPlugin):
    """Authentication plugin for Microsoft ADFS2.0 IdPs.

    :param auth_url: URL of the Identity Service
    :type auth_url: string

    :param identity_provider: name of the Identity Provider the client will
                              authenticate against. This parameter will be used
                              to build a dynamic URL used to obtain unscoped
                              OpenStack token.
    :type identity_provider: string

    :param identity_provider_url: An Identity Provider URL, where the SAML2
                                  authentication request will be sent.
    :type identity_provider_url: string

    :param service_provider_endpoint: Endpoint where an assertion is being
        sent, for instance: ``https://host.domain/Shibboleth.sso/ADFS``
    :type service_provider_endpoint: string

    :param username: User's login
    :type username: string

    :param password: User's password
    :type password: string

    """
    _auth_method_class = Saml2UnscopedTokenAuthMethod
    DEFAULT_ADFS_TOKEN_EXPIRATION = 120
    HEADER_SOAP = {'Content-Type': 'application/soap+xml; charset=utf-8'}
    HEADER_X_FORM = {'Content-Type': 'application/x-www-form-urlencoded'}
    NAMESPACES = {'s': 'http://www.w3.org/2003/05/soap-envelope', 'a': 'http://www.w3.org/2005/08/addressing', 'u': 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd'}
    ADFS_TOKEN_NAMESPACES = {'s': 'http://www.w3.org/2003/05/soap-envelope', 't': 'http://docs.oasis-open.org/ws-sx/ws-trust/200512'}
    ADFS_ASSERTION_XPATH = '/s:Envelope/s:Body/t:RequestSecurityTokenResponseCollection/t:RequestSecurityTokenResponse'

    def __init__(self, auth_url, identity_provider, identity_provider_url, service_provider_endpoint, username, password, **kwargs):
        super(ADFSUnscopedToken, self).__init__(auth_url=auth_url, **kwargs)
        self.identity_provider = identity_provider
        self.identity_provider_url = identity_provider_url
        self.service_provider_endpoint = service_provider_endpoint
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

    @classmethod
    def get_options(cls):
        options = super(ADFSUnscopedToken, cls).get_options()
        options.extend([cfg.StrOpt('service-provider-endpoint', help="Service Provider's Endpoint")])
        return options

    def _cookies(self, session):
        """Check if cookie jar is not empty.

        keystoneclient.session.Session object doesn't have a cookies attribute.
        We should then try fetching cookies from the underlying
        requests.Session object. If that fails too, there is something wrong
        and let Python raise the AttributeError.

        :param session
        :returns: True if cookie jar is nonempty, False otherwise
        :raises AttributeError: in case cookies are not find anywhere

        """
        try:
            return bool(session.cookies)
        except AttributeError:
            pass
        return bool(session.session.cookies)

    def _token_dates(self, fmt='%Y-%m-%dT%H:%M:%S.%fZ'):
        """Calculate created and expires datetime objects.

        The method is going to be used for building ADFS Request Security
        Token message. Time interval between ``created`` and ``expires``
        dates is now static and equals to 120 seconds. ADFS security tokens
        should not be live too long, as currently ``keystoneclient``
        doesn't have mechanisms for reusing such tokens (every time ADFS authn
        method is called, keystoneclient will login with the ADFS instance).

        :param fmt: Datetime format for specifying string format of a date.
                    It should not be changed if the method is going to be used
                    for building the ADFS security token request.
        :type fmt: string

        """
        date_created = datetime.datetime.utcnow()
        date_expires = date_created + datetime.timedelta(seconds=self.DEFAULT_ADFS_TOKEN_EXPIRATION)
        return [_time.strftime(fmt) for _time in (date_created, date_expires)]

    def _prepare_adfs_request(self):
        """Build the ADFS Request Security Token SOAP message.

        Some values like username or password are inserted in the request.

        """
        WSS_SECURITY_NAMESPACE = {'o': 'http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd'}
        TRUST_NAMESPACE = {'trust': 'http://docs.oasis-open.org/ws-sx/ws-trust/200512'}
        WSP_NAMESPACE = {'wsp': 'http://schemas.xmlsoap.org/ws/2004/09/policy'}
        WSA_NAMESPACE = {'wsa': 'http://www.w3.org/2005/08/addressing'}
        root = etree.Element('{http://www.w3.org/2003/05/soap-envelope}Envelope', nsmap=self.NAMESPACES)
        header = etree.SubElement(root, '{http://www.w3.org/2003/05/soap-envelope}Header')
        action = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}Action')
        action.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
        action.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/RST/Issue'
        messageID = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}MessageID')
        messageID.text = 'urn:uuid:' + uuid.uuid4().hex
        replyID = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}ReplyTo')
        address = etree.SubElement(replyID, '{http://www.w3.org/2005/08/addressing}Address')
        address.text = 'http://www.w3.org/2005/08/addressing/anonymous'
        to = etree.SubElement(header, '{http://www.w3.org/2005/08/addressing}To')
        to.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
        security = etree.SubElement(header, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Security', nsmap=WSS_SECURITY_NAMESPACE)
        security.set('{http://www.w3.org/2003/05/soap-envelope}mustUnderstand', '1')
        timestamp = etree.SubElement(security, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Timestamp')
        timestamp.set('{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Id', '_0')
        created = etree.SubElement(timestamp, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Created')
        expires = etree.SubElement(timestamp, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}Expires')
        created.text, expires.text = self._token_dates()
        usernametoken = etree.SubElement(security, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}UsernameToken')
        usernametoken.set('{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd}u', 'uuid-%s-1' % uuid.uuid4().hex)
        username = etree.SubElement(usernametoken, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Username')
        password = etree.SubElement(usernametoken, '{http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd}Password', Type='http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText')
        body = etree.SubElement(root, '{http://www.w3.org/2003/05/soap-envelope}Body')
        request_security_token = etree.SubElement(body, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}RequestSecurityToken', nsmap=TRUST_NAMESPACE)
        applies_to = etree.SubElement(request_security_token, '{http://schemas.xmlsoap.org/ws/2004/09/policy}AppliesTo', nsmap=WSP_NAMESPACE)
        endpoint_reference = etree.SubElement(applies_to, '{http://www.w3.org/2005/08/addressing}EndpointReference', nsmap=WSA_NAMESPACE)
        wsa_address = etree.SubElement(endpoint_reference, '{http://www.w3.org/2005/08/addressing}Address')
        keytype = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}KeyType')
        keytype.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Bearer'
        request_type = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}RequestType')
        request_type.text = 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/Issue'
        token_type = etree.SubElement(request_security_token, '{http://docs.oasis-open.org/ws-sx/ws-trust/200512}TokenType')
        token_type.text = 'urn:oasis:names:tc:SAML:1.0:assertion'
        username.text = self.username
        password.text = self.password
        to.text = self.identity_provider_url
        wsa_address.text = self.service_provider_endpoint
        self.prepared_request = root

    def _get_adfs_security_token(self, session):
        """Send ADFS Security token to the ADFS server.

        Store the result in the instance attribute and raise an exception in
        case the response is not valid XML data.

        If a user cannot authenticate due to providing bad credentials, the
        ADFS2.0 server will return a HTTP 500 response and a XML Fault message.
        If ``exceptions.InternalServerError`` is caught, the method tries to
        parse the XML response.
        If parsing is unsuccessful, an ``exceptions.AuthorizationFailure`` is
        raised with a reason from the XML fault. Otherwise an original
        ``exceptions.InternalServerError`` is re-raised.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :raises keystoneclient.exceptions.AuthorizationFailure: when HTTP
                 response from the ADFS server is not a valid XML ADFS security
                 token.
        :raises keystoneclient.exceptions.InternalServerError: If response
                 status code is HTTP 500 and the response XML cannot be
                 recognized.

        """

        def _get_failure(e):
            xpath = '/s:Envelope/s:Body/s:Fault/s:Code/s:Subcode/s:Value'
            content = e.response.content
            try:
                obj = self.str_to_xml(content).xpath(xpath, namespaces=self.NAMESPACES)
                obj = self._first(obj)
                return obj.text
            except (IndexError, exceptions.AuthorizationFailure):
                raise e
        request_security_token = self.xml_to_str(self.prepared_request)
        try:
            response = session.post(url=self.identity_provider_url, headers=self.HEADER_SOAP, data=request_security_token, authenticated=False)
        except exceptions.InternalServerError as e:
            reason = _get_failure(e)
            raise exceptions.AuthorizationFailure(reason)
        msg = _('Error parsing XML returned from the ADFS Identity Provider, reason: %s')
        self.adfs_token = self.str_to_xml(response.content, msg)

    def _prepare_sp_request(self):
        """Prepare ADFS Security Token to be sent to the Service Provider.

        The method works as follows:
        * Extract SAML2 assertion from the ADFS Security Token.
        * Replace namespaces
        * urlencode assertion
        * concatenate static string with the encoded assertion

        """
        assertion = self.adfs_token.xpath(self.ADFS_ASSERTION_XPATH, namespaces=self.ADFS_TOKEN_NAMESPACES)
        assertion = self._first(assertion)
        assertion = self.xml_to_str(assertion)
        assertion = assertion.replace(b'http://docs.oasis-open.org/ws-sx/ws-trust/200512', b'http://schemas.xmlsoap.org/ws/2005/02/trust')
        encoded_assertion = urllib.parse.quote(assertion)
        self.encoded_assertion = 'wa=wsignin1.0&wresult=' + encoded_assertion

    def _send_assertion_to_service_provider(self, session):
        """Send prepared assertion to a service provider.

        As the assertion doesn't contain a protected resource, the value from
        the ``location`` header is not valid and we should not let the Session
        object get redirected there. The aim of this call is to get a cookie in
        the response which is required for entering a protected endpoint.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :raises: Corresponding HTTP error exception

        """
        session.post(url=self.service_provider_endpoint, data=self.encoded_assertion, headers=self.HEADER_X_FORM, redirect=False, authenticated=False)

    def _access_service_provider(self, session):
        """Access protected endpoint and fetch unscoped token.

        After federated authentication workflow a protected endpoint should be
        accessible with the session object. The access is granted basing on the
        cookies stored within the session object. If, for some reason no
        cookies are present (quantity test) it means something went wrong and
        user will not be able to fetch an unscoped token. In that case an
        ``exceptions.AuthorizationFailure` exception is raised and no HTTP call
        is even made.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :raises keystoneclient.exceptions.AuthorizationFailure: in case session
        object has empty cookie jar.

        """
        if self._cookies(session) is False:
            raise exceptions.AuthorizationFailure(_("Session object doesn't contain a cookie, therefore you are not allowed to enter the Identity Provider's protected area."))
        self.authenticated_response = session.get(self.token_url, authenticated=False)

    def _get_unscoped_token(self, session, *kwargs):
        """Retrieve unscoped token after authentcation with ADFS server.

        This is a multistep process::

        * Prepare ADFS Request Securty Token -
        build an etree.XML object filling certain attributes with proper user
        credentials, created/expires dates (ticket is be valid for 120 seconds
        as currently we don't handle reusing ADFS issued security tokens) .
        Step handled by ``ADFSUnscopedToken._prepare_adfs_request()`` method.

        * Send ADFS Security token to the ADFS server. Step handled by
        ``ADFSUnscopedToken._get_adfs_security_token()`` method.

        * Receive and parse security token, extract actual SAML assertion and
        prepare a request addressed for the Service Provider endpoint.
        This also includes changing namespaces in the XML document. Step
        handled by ``ADFSUnscopedToken._prepare_sp_request()`` method.

        * Send prepared assertion to the Service Provider endpoint. Usually
        the server will respond with HTTP 301 code which should be ignored as
        the 'location' header doesn't contain protected area. The goal of this
        operation is fetching the session cookie which later allows for
        accessing protected URL endpoints. Step handed by
        ``ADFSUnscopedToken._send_assertion_to_service_provider()`` method.

        * Once the session cookie is issued, the protected endpoint can be
        accessed and an unscoped token can be retrieved. Step handled by
        ``ADFSUnscopedToken._access_service_provider()`` method.

        :param session : a session object to send out HTTP requests.
        :type session: keystoneclient.session.Session

        :returns: (Unscoped federated token, token JSON body)

        """
        self._prepare_adfs_request()
        self._get_adfs_security_token(session)
        self._prepare_sp_request()
        self._send_assertion_to_service_provider(session)
        self._access_service_provider(session)
        try:
            return (self.authenticated_response.headers['X-Subject-Token'], self.authenticated_response.json()['token'])
        except (KeyError, ValueError):
            raise exceptions.InvalidResponse(response=self.authenticated_response)

    def get_auth_ref(self, session, **kwargs):
        token, token_json = self._get_unscoped_token(session)
        return access.AccessInfoV3(token, **token_json)