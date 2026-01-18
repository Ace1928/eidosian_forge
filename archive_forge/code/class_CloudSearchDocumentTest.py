from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchDocumentTest(unittest.TestCase):

    def setUp(self):
        HTTPretty.enable()
        HTTPretty.register_uri(HTTPretty.POST, 'http://doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com/2011-02-01/documents/batch', body=json.dumps(self.response).encode('utf-8'), content_type='application/json')

    def tearDown(self):
        HTTPretty.disable()