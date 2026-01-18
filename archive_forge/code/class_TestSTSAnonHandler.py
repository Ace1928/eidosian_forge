import copy
from mock import Mock
from tests.unit import unittest
from boto.auth import STSAnonHandler
from boto.connection import HTTPRequest
class TestSTSAnonHandler(unittest.TestCase):

    def setUp(self):
        self.provider = Mock()
        self.provider.access_key = 'access_key'
        self.provider.secret_key = 'secret_key'
        self.request = HTTPRequest(method='GET', protocol='https', host='sts.amazonaws.com', port=443, path='/', auth_path=None, params={'Action': 'AssumeRoleWithWebIdentity', 'Version': '2011-06-15', 'RoleSessionName': 'web-identity-federation', 'ProviderId': '2012-06-01', 'WebIdentityToken': 'Atza|IQEBLjAsAhRkcxQ'}, headers={}, body='')

    def test_escape_value(self):
        auth = STSAnonHandler('sts.amazonaws.com', Mock(), self.provider)
        value = auth._escape_value('Atza|IQEBLjAsAhRkcxQ')
        self.assertEqual(value, 'Atza%7CIQEBLjAsAhRkcxQ')

    def test_build_query_string(self):
        auth = STSAnonHandler('sts.amazonaws.com', Mock(), self.provider)
        query_string = auth._build_query_string(self.request.params)
        self.assertEqual(query_string, 'Action=AssumeRoleWithWebIdentity' + '&ProviderId=2012-06-01&RoleSessionName=web-identity-federation' + '&Version=2011-06-15&WebIdentityToken=Atza%7CIQEBLjAsAhRkcxQ')

    def test_add_auth(self):
        auth = STSAnonHandler('sts.amazonaws.com', Mock(), self.provider)
        req = copy.copy(self.request)
        auth.add_auth(req)
        self.assertEqual(req.body, 'Action=AssumeRoleWithWebIdentity' + '&ProviderId=2012-06-01&RoleSessionName=web-identity-federation' + '&Version=2011-06-15&WebIdentityToken=Atza%7CIQEBLjAsAhRkcxQ')