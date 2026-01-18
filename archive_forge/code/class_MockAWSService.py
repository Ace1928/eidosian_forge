import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
class MockAWSService(AWSQueryConnection):
    """
    Fake AWS Service

    This is used to test the AWSQueryConnection object is behaving properly.
    """
    APIVersion = '2012-01-01'

    def _required_auth_capability(self):
        return ['sign-v2']

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, is_secure=True, host=None, port=None, proxy=None, proxy_port=None, proxy_user=None, proxy_pass=None, debug=0, https_connection_factory=None, region=None, path='/', api_version=None, security_token=None, validate_certs=True, profile_name=None):
        self.region = region
        if host is None:
            host = self.region.endpoint
        AWSQueryConnection.__init__(self, aws_access_key_id, aws_secret_access_key, is_secure, port, proxy, proxy_port, proxy_user, proxy_pass, host, debug, https_connection_factory, path, security_token, validate_certs=validate_certs, profile_name=profile_name)