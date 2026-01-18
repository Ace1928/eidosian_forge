import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.common.google import GoogleResponse, GoogleBaseConnection, ResourceNotFoundError
class GoogleDNSConnection(GoogleBaseConnection):
    host = 'www.googleapis.com'
    responseCls = GoogleDNSResponse

    def __init__(self, user_id, key, secure, auth_type=None, credential_file=None, project=None, **kwargs):
        super().__init__(user_id, key, secure=secure, auth_type=auth_type, credential_file=credential_file, **kwargs)
        self.request_path = '/dns/{}/projects/{}'.format(API_VERSION, project)