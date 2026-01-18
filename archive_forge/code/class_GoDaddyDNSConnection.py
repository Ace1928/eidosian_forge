from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyDNSConnection(ConnectionKey):
    responseCls = GoDaddyDNSResponse
    host = API_HOST
    allow_insecure = False

    def __init__(self, key, secret, secure=True, shopper_id=None, host=None, port=None, url=None, timeout=None, proxy_url=None, backoff=None, retry_delay=None):
        super().__init__(key, secure=secure, host=host, port=port, url=url, timeout=timeout, proxy_url=proxy_url, backoff=backoff, retry_delay=retry_delay)
        self.key = key
        self.secret = secret
        self.shopper_id = shopper_id

    def add_default_headers(self, headers):
        if self.shopper_id is not None:
            headers['X-Shopper-Id'] = self.shopper_id
        headers['Content-type'] = 'application/json'
        headers['Authorization'] = 'sso-key {}:{}'.format(self.key, self.secret)
        return headers