import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestRemoveTag(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'<RemoveTagsResponse\n               xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n                   <RemoveTagsResult/>\n                   <ResponseMetadata>\n                        <RequestId>88888888-8888-8888-8888-888888888888</RequestId>\n                   </ResponseMetadata>\n               </RemoveTagsResponse>\n               '

    def test_remove_tags(self):
        input_tags = {'FirstKey': 'One', 'SecondKey': 'Two', 'ZzzNoValue': ''}
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.add_tags()
        with self.assertRaises(TypeError):
            self.service_connection.add_tags('j-123')
        with self.assertRaises(AssertionError):
            self.service_connection.add_tags('j-123', [])
        response = self.service_connection.remove_tags('j-123', ['FirstKey', 'SecondKey'])
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'RemoveTags', 'ResourceId': 'j-123', 'TagKeys.member.1': 'FirstKey', 'TagKeys.member.2': 'SecondKey', 'Version': '2009-03-31'})