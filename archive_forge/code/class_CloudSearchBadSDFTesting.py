from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchBadSDFTesting(CloudSearchDocumentTest):
    response = {'status': 'success', 'adds': 1, 'deletes': 0}

    def test_cloudsearch_erroneous_sdf(self):
        original = boto.log.error
        boto.log.error = MagicMock()
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.add('1234', 10, {'id': '1234', 'title': None, 'category': ['cat_a', 'cat_b', 'cat_c']})
        document.commit()
        self.assertNotEqual(len(boto.log.error.call_args_list), 1)
        boto.log.error = original