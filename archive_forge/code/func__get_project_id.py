import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
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