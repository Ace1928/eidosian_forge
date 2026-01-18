import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _fetch_auth_token(self):
    """
        Fetch our authentication token and service catalog.
        """
    headers = {'X-Subject-Token': self.auth_token}
    response = self.authenticated_request('/v3/auth/tokens', headers=headers)
    self._parse_token_response(response)
    return self