from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.tagging import Tag
class TestS3Tagging(AWSMockServiceTestCase):
    connection_class = S3Connection

    def default_body(self):
        return '\n            <Tagging>\n              <TagSet>\n                 <Tag>\n                   <Key>Project</Key>\n                   <Value>Project One</Value>\n                 </Tag>\n                 <Tag>\n                   <Key>User</Key>\n                   <Value>jsmith</Value>\n                 </Tag>\n              </TagSet>\n            </Tagging>\n        '

    def test_parse_tagging_response(self):
        self.set_http_response(status_code=200)
        b = Bucket(self.service_connection, 'mybucket')
        api_response = b.get_tags()
        self.assertEqual(len(api_response), 1)
        self.assertEqual(len(api_response[0]), 2)
        self.assertEqual(api_response[0][0].key, 'Project')
        self.assertEqual(api_response[0][0].value, 'Project One')
        self.assertEqual(api_response[0][1].key, 'User')
        self.assertEqual(api_response[0][1].value, 'jsmith')

    def test_tag_equality(self):
        t1 = Tag('foo', 'bar')
        t2 = Tag('foo', 'bar')
        t3 = Tag('foo', 'baz')
        t4 = Tag('baz', 'bar')
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(t1, t4)