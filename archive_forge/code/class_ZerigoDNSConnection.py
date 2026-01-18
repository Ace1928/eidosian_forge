import copy
import base64
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib
from libcloud.utils.xml import findall, findtext
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
class ZerigoDNSConnection(ConnectionUserAndKey):
    host = API_HOST
    secure = True
    responseCls = ZerigoDNSResponse

    def add_default_headers(self, headers):
        auth_b64 = base64.b64encode(b('{}:{}'.format(self.user_id, self.key)))
        headers['Authorization'] = 'Basic %s' % auth_b64.decode('utf-8')
        return headers

    def request(self, action, params=None, data='', headers=None, method='GET'):
        if not headers:
            headers = {}
        if not params:
            params = {}
        if method in ('POST', 'PUT'):
            headers = {'Content-Type': 'application/xml; charset=UTF-8'}
        return super().request(action=action, params=params, data=data, method=method, headers=headers)