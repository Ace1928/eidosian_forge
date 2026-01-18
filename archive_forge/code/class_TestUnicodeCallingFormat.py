from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
class TestUnicodeCallingFormat(AWSMockServiceTestCase):
    connection_class = S3Connection

    def default_body(self):
        return '<?xml version="1.0" encoding="UTF-8"?>\n<ListAllMyBucketsResult xmlns="http://doc.s3.amazonaws.com/2006-03-01">\n  <Owner>\n    <ID>bcaf1ffd86f461ca5fb16fd081034f</ID>\n    <DisplayName>webfile</DisplayName>\n  </Owner>\n  <Buckets>\n    <Bucket>\n      <Name>quotes</Name>\n      <CreationDate>2006-02-03T16:45:09.000Z</CreationDate>\n    </Bucket>\n    <Bucket>\n      <Name>samples</Name>\n      <CreationDate>2006-02-03T16:41:58.000Z</CreationDate>\n    </Bucket>\n  </Buckets>\n</ListAllMyBucketsResult>'

    def create_service_connection(self, **kwargs):
        kwargs['calling_format'] = u'boto.s3.connection.OrdinaryCallingFormat'
        return super(TestUnicodeCallingFormat, self).create_service_connection(**kwargs)

    def test_unicode_calling_format(self):
        self.set_http_response(status_code=200)
        self.service_connection.get_all_buckets()