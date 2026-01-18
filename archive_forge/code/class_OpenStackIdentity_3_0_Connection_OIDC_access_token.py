import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_3_0_Connection_OIDC_access_token(OpenStackIdentity_3_0_Connection):
    """
    Connection class for Keystone API v3.x. using OpenID Connect tokens

    The OIDC token must be set in the self.key attribute.

    The identity provider name required to get the full path
    must be set in the self.user_id attribute.

    The protocol name required to get the full path
    must be set in the self.tenant_name attribute.

    The self.domain_name attribute can be used either to select the
    domain name in case of domain scoped token or to select the project
    name in case of project scoped token
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v3.x with OIDC support'
    auth_version = '3.0'

    def authenticate(self, force=False):
        """
        Perform authentication.
        """
        if not self._is_authentication_needed(force=force):
            return self
        subject_token = self._get_unscoped_token_from_oidc_token()
        data = {'auth': {'identity': {'methods': ['token'], 'token': {'id': subject_token}}}}
        if self.token_scope == OpenStackIdentityTokenScope.PROJECT:
            project_id = self._get_project_id(token=subject_token)
            data['auth']['scope'] = {'project': {'id': project_id}}
        elif self.token_scope == OpenStackIdentityTokenScope.DOMAIN:
            data['auth']['scope'] = {'domain': {'name': self.domain_name}}
        elif self.token_scope == OpenStackIdentityTokenScope.UNSCOPED:
            pass
        else:
            raise ValueError('Token needs to be scoped either to project or a domain')
        data = json.dumps(data)
        response = self.request('/v3/auth/tokens', data=data, headers={'Content-Type': 'application/json'}, method='POST')
        self._parse_token_response(response, cache_it=True, raise_ambiguous_version_error=False)
        return self

    def _get_unscoped_token_from_oidc_token(self):
        """
        Get unscoped token from OIDC access token
        """
        path = '/v3/OS-FEDERATION/identity_providers/{}/protocols/{}/auth'.format(self.user_id, self.tenant_name)
        response = self.request(path, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % self.key}, method='GET')
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif response.status in [httplib.OK, httplib.CREATED]:
            if 'x-subject-token' in response.headers:
                return response.headers['x-subject-token']
            else:
                raise MalformedResponseError('No x-subject-token returned', driver=self.driver)
        else:
            raise MalformedResponseError('Malformed response', driver=self.driver, body=response.body)

    def _get_project_id(self, token):
        """
        Get the first project ID accessible with the specified access token
        """
        path = '/v3/auth/projects'
        response = self.request(path, headers={'Content-Type': 'application/json', AUTH_TOKEN_HEADER: token}, method='GET')
        if response.status not in [httplib.UNAUTHORIZED, httplib.OK, httplib.CREATED]:
            path = '/v3/OS-FEDERATION/projects'
            response = self.request(path, headers={'Content-Type': 'application/json', AUTH_TOKEN_HEADER: token}, method='GET')
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif response.status in [httplib.OK, httplib.CREATED]:
            try:
                body = json.loads(response.body)
                if self.domain_name and self.domain_name != 'Default':
                    for project in body['projects']:
                        if self.domain_name in [project['name'], project['id']]:
                            return project['id']
                    raise ValueError('Project %s not found' % self.domain_name)
                else:
                    return body['projects'][0]['id']
            except ValueError as e:
                raise e
            except Exception as e:
                raise MalformedResponseError('Failed to parse JSON', e)
        else:
            raise MalformedResponseError('Malformed response', driver=self.driver, body=response.body)