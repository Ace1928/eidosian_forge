import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _get_tenant_name(self, token):
    """
        Get the first available tenant name (usually there are only one)
        """
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json', AUTH_TOKEN_HEADER: token}
    response = self.request('/v2.0/tenants', headers=headers, method='GET')
    if response.status == httplib.UNAUTHORIZED:
        raise InvalidCredsError()
    elif response.status in [httplib.OK, httplib.CREATED]:
        try:
            body = json.loads(response.body)
            return body['tenants'][0]['name']
        except Exception as e:
            raise MalformedResponseError('Failed to parse JSON', e)
    else:
        raise MalformedResponseError('Malformed response', driver=self.driver, body=response.body)