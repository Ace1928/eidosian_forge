import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def _v2_auth(self, token_url):
    creds = self.creds
    creds = {'auth': {'tenantName': creds['tenant'], 'passwordCredentials': {'username': creds['username'], 'password': creds['password']}}}
    headers = {'Content-Type': 'application/json'}
    req_body = jsonutils.dumps(creds)
    resp, resp_body = self._do_request(token_url, 'POST', headers=headers, body=req_body)
    if resp.status == http.OK:
        resp_auth = jsonutils.loads(resp_body)['access']
        creds_region = self.creds.get('region')
        if self.configure_via_auth:
            endpoint = get_endpoint(resp_auth['serviceCatalog'], endpoint_region=creds_region)
            self.management_url = endpoint
        self.auth_token = resp_auth['token']['id']
    elif resp.status == http.USE_PROXY:
        raise exception.RedirectException(resp['location'])
    elif resp.status == http.BAD_REQUEST:
        raise exception.AuthBadRequest(url=token_url)
    elif resp.status == http.UNAUTHORIZED:
        raise exception.NotAuthenticated()
    elif resp.status == http.NOT_FOUND:
        raise exception.AuthUrlNotFound(url=token_url)
    else:
        raise Exception(_('Unexpected response: %s') % resp.status)