import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _authenticate_2_0_with_api_key(self):
    data = {'auth': {'RAX-KSKEY:apiKeyCredentials': {'username': self.user_id, 'apiKey': self.key}}}
    if self.tenant_name:
        data['auth']['tenantName'] = self.tenant_name
    reqbody = json.dumps(data)
    return self._authenticate_2_0_with_body(reqbody)