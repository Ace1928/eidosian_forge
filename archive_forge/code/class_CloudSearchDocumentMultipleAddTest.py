from tests.unit import unittest
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.document import CommitMismatchError, EncodingError, \
import boto
class CloudSearchDocumentMultipleAddTest(CloudSearchDocumentTest):
    response = {'status': 'success', 'adds': 3, 'deletes': 0}
    objs = {'1234': {'version': 10, 'fields': {'id': '1234', 'title': 'Title 1', 'category': ['cat_a', 'cat_b', 'cat_c']}}, '1235': {'version': 11, 'fields': {'id': '1235', 'title': 'Title 2', 'category': ['cat_b', 'cat_c', 'cat_d']}}, '1236': {'version': 12, 'fields': {'id': '1236', 'title': 'Title 3', 'category': ['cat_e', 'cat_f', 'cat_g']}}}

    def test_cloudsearch_add_basics(self):
        """Check that multiple documents are added correctly to AWS"""
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        for key, obj in self.objs.items():
            document.add(key, obj['version'], obj['fields'])
        document.commit()
        args = json.loads(HTTPretty.last_request.body.decode('utf-8'))
        for arg in args:
            self.assertTrue(arg['id'] in self.objs)
            self.assertEqual(arg['version'], self.objs[arg['id']]['version'])
            self.assertEqual(arg['fields']['id'], self.objs[arg['id']]['fields']['id'])
            self.assertEqual(arg['fields']['title'], self.objs[arg['id']]['fields']['title'])
            self.assertEqual(arg['fields']['category'], self.objs[arg['id']]['fields']['category'])

    def test_cloudsearch_add_results(self):
        """
        Check that the result from adding multiple documents is parsed
        correctly.
        """
        document = DocumentServiceConnection(endpoint='doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
        for key, obj in self.objs.items():
            document.add(key, obj['version'], obj['fields'])
        doc = document.commit()
        self.assertEqual(doc.status, 'success')
        self.assertEqual(doc.adds, len(self.objs))
        self.assertEqual(doc.deletes, 0)