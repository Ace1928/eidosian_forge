import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
class _SamlAuth(requests.auth.AuthBase):
    """A generic SAML ECP plugin for requests.

    This is a multi-step process including multiple HTTP requests.
    Authentication consists of:

    * HTTP GET request to the Service Provider.

        It's crucial to include HTTP headers indicating we are expecting SOAP
        message in return. Service Provider should respond with a SOAP
        message.

    * HTTP POST request to the external Identity Provider service with
        ECP extension enabled. The content sent is a header removed SOAP
        message returned from the Service Provider. It's also worth noting
        that ECP extension to the SAML2 doesn't define authentication method.
        The most popular is HttpBasicAuth with just user and password.
        Other possibilities could be X509 certificates or Kerberos.
        Upon successful authentication the user should receive a SAML2
        assertion.

    * HTTP POST request again to the Service Provider. The body of the
        request includes SAML2 assertion issued by a trusted Identity
        Provider. The request should be sent to the Service Provider
        consumer url specified in the SAML2 assertion.
        Providing the authentication was successful and both Service Provider
        and Identity Providers are trusted to each other, the Service
        Provider will issue an unscoped token with a list of groups the
        federated user is a member of.
    """

    def __init__(self, identity_provider_url, requests_auth):
        super(_SamlAuth, self).__init__()
        self.identity_provider_url = identity_provider_url
        self.requests_auth = requests_auth

    def __call__(self, request):
        try:
            accept = request.headers['Accept']
        except KeyError:
            request.headers['Accept'] = _PAOS_HEADER
        else:
            request.headers['Accept'] = ','.join([accept, _PAOS_HEADER])
        request.headers['PAOS'] = _PAOS_VER
        request.register_hook('response', self._handle_response)
        return request

    def _handle_response(self, response, **kwargs):
        if response.status_code == 200 and response.headers.get('Content-Type') == _PAOS_HEADER:
            response = self._ecp_retry(response, **kwargs)
        return response

    def _ecp_retry(self, sp_response, **kwargs):
        history = [sp_response]

        def send(*send_args, **send_kwargs):
            req = requests.Request(*send_args, **send_kwargs)
            return sp_response.connection.send(req.prepare(), **kwargs)
        authn_request = _response_xml(sp_response, 'Service Provider')
        relay_state = _str_from_xml(authn_request, _XPATH_SP_RELAY_STATE)
        sp_consumer_url = _str_from_xml(authn_request, _XPATH_SP_CONSUMER_URL)
        authn_request.remove(authn_request[0])
        idp_response = send('POST', self.identity_provider_url, headers={'Content-type': 'text/xml'}, data=etree.tostring(authn_request), auth=self.requests_auth)
        history.append(idp_response)
        authn_response = _response_xml(idp_response, 'Identity Provider')
        idp_consumer_url = _str_from_xml(authn_response, _XPATH_IDP_CONSUMER_URL)
        if sp_consumer_url != idp_consumer_url:
            send('POST', sp_consumer_url, data=_SOAP_FAULT, headers={'Content-Type': _PAOS_HEADER})
            msg = 'Consumer URLs from Service Provider %(service_provider)s %(sp_consumer_url)s and Identity Provider %(identity_provider)s %(idp_consumer_url)s are not equal'
            msg = msg % {'service_provider': sp_response.request.url, 'sp_consumer_url': sp_consumer_url, 'identity_provider': self.identity_provider_url, 'idp_consumer_url': idp_consumer_url}
            raise ConsumerMismatch(msg)
        authn_response[0][0] = relay_state
        final_resp = send('POST', idp_consumer_url, headers={'Content-Type': _PAOS_HEADER}, cookies=idp_response.cookies, data=etree.tostring(authn_response))
        history.append(final_resp)
        if final_resp.status_code in (requests.codes.found, requests.codes.other):
            sp_response.content
            sp_response.raw.release_conn()
            req = sp_response.request.copy()
            req.url = final_resp.headers['location']
            req.prepare_cookies(final_resp.cookies)
            final_resp = sp_response.connection.send(req, **kwargs)
            history.append(final_resp)
        final_resp.history.extend(history)
        return final_resp