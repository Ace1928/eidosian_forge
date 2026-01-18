from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchDocumentDelete(CloudSearchDocumentTest):
    response = {'status': 'success', 'adds': 0, 'deletes': 1}

    def test_cloudsearch_delete(self):
        """
        Test that the request for a single document deletion is done properly.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.delete('5', '10')
        document.commit()
        args = json.loads(HTTPretty.last_request.body.decode('utf-8'))[0]
        self.assertEqual(args['version'], '10')
        self.assertEqual(args['type'], 'delete')
        self.assertEqual(args['id'], '5')

    def test_cloudsearch_delete_results(self):
        """
        Check that the result of a single document deletion is parsed properly.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        document.delete('5', '10')
        doc = document.commit()
        self.assertEqual(doc.status, 'success')
        self.assertEqual(doc.adds, 0)
        self.assertEqual(doc.deletes, 1)