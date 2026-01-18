import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def _v3_auth(self, token_url):
    creds = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.creds['username'], 'domain': {'id': self.creds['user_domain_id']}, 'password': self.creds['password']}}}, 'scope': {'project': {'name': self.creds['project'], 'domain': {'id': self.creds['project_domain_id']}}}}}
    headers = {'Content-Type': 'application/json'}
    req_body = jsonutils.dumps(creds)
    resp, resp_body = self._do_request(token_url, 'POST', headers=headers, body=req_body)
    resp_body = jsonutils.loads(resp_body)
    if resp.status == 201:
        resp_auth = resp['x-subject-token']
        creds_region = self.creds.get('region')
        if self.configure_via_auth:
            endpoint = get_endpoint(resp_body['token']['catalog'], endpoint_region=creds_region)
            self.management_url = endpoint
        self.auth_token = resp_auth
    elif resp.status == 305:
        raise exception.RedirectException(resp['location'])
    elif resp.status == 400:
        raise exception.AuthBadRequest(url=token_url)
    elif resp.status == 401:
        raise Exception(_('Unexpected response: %s') % resp.status)