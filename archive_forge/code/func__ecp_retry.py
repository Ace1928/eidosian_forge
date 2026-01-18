import abc
import requests
import requests.auth
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import v3
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