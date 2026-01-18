import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
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