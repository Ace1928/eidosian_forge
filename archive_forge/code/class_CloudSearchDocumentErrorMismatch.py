from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchDocumentErrorMismatch(CloudSearchDocumentTest):
    response = {'status': 'error', 'adds': 0, 'deletes': 0, 'errors': [{'message': 'Something went wrong'}]}

    def test_fake_failure(self):
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']})
        self.assertRaises(CommitMismatchError, document.commit)