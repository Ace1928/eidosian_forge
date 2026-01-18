import io
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.compat import StringIO
from boto.exception import BotoServerError
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.key import Key
class TestS3Key(AWSMockServiceTestCase):
    connection_class = S3Connection

    def setUp(self):
        super(TestS3Key, self).setUp()

    def default_body(self):
        return 'default body'

    def test_unicode_name(self):
        k = Key()
        k.name = u'Österreich'
        print(repr(k))

    def test_when_no_restore_header_present(self):
        self.set_http_response(status_code=200)
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('myglacierkey')
        self.assertIsNone(k.ongoing_restore)
        self.assertIsNone(k.expiry_date)

    def test_restore_header_with_ongoing_restore(self):
        self.set_http_response(status_code=200, header=[('x-amz-restore', 'ongoing-request="true"')])
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('myglacierkey')
        self.assertTrue(k.ongoing_restore)
        self.assertIsNone(k.expiry_date)

    def test_restore_completed(self):
        self.set_http_response(status_code=200, header=[('x-amz-restore', 'ongoing-request="false", expiry-date="Fri, 21 Dec 2012 00:00:00 GMT"')])
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('myglacierkey')
        self.assertFalse(k.ongoing_restore)
        self.assertEqual(k.expiry_date, 'Fri, 21 Dec 2012 00:00:00 GMT')

    def test_delete_key_return_key(self):
        self.set_http_response(status_code=204, body='')
        b = Bucket(self.service_connection, 'mybucket')
        key = b.delete_key('fookey')
        self.assertIsNotNone(key)

    def test_storage_class(self):
        self.set_http_response(status_code=200)
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('fookey')
        k.bucket = mock.MagicMock()
        k.set_contents_from_string('test')
        k.bucket.list.assert_not_called()
        sc_value = k.storage_class
        self.assertEqual(sc_value, 'STANDARD')
        k.bucket.list.assert_called_with(k.name.encode('utf-8'))
        k.bucket.list.reset_mock()
        k.storage_class = 'GLACIER'
        k.set_contents_from_string('test')
        k.bucket.list.assert_not_called()

    def test_change_storage_class(self):
        self.set_http_response(status_code=200)
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('fookey')
        k.copy = mock.MagicMock()
        k.bucket = mock.MagicMock()
        k.bucket.name = 'mybucket'
        self.assertEqual(k.storage_class, 'STANDARD')
        k.change_storage_class('REDUCED_REDUNDANCY')
        k.copy.assert_called_with('mybucket', 'fookey', reduced_redundancy=True, preserve_acl=True, validate_dst_bucket=True)

    def test_change_storage_class_new_bucket(self):
        self.set_http_response(status_code=200)
        b = Bucket(self.service_connection, 'mybucket')
        k = b.get_key('fookey')
        k.copy = mock.MagicMock()
        k.bucket = mock.MagicMock()
        k.bucket.name = 'mybucket'
        self.assertEqual(k.storage_class, 'STANDARD')
        k.copy.reset_mock()
        k.change_storage_class('REDUCED_REDUNDANCY', dst_bucket='yourbucket')
        k.copy.assert_called_with('yourbucket', 'fookey', reduced_redundancy=True, preserve_acl=True, validate_dst_bucket=True)

    def test_download_succeeds(self):
        test_case_headers = [[('Content-Length', '5')], [('Content-Range', 'bytes 15-19/100')]]
        for headers in test_case_headers:
            with self.subTest(headers=headers):
                head_object_response = self.create_response(status_code=200, header=headers)
                media_response = self.create_response(status_code=200, header=headers)
                media_response.read.side_effect = [b'12345', b'', b'']
                self.https_connection.getresponse.side_effect = [head_object_response, media_response]
                bucket = Bucket(self.service_connection, 'bucket')
                key = bucket.get_key('object')
                output_stream = io.BytesIO()
                key.get_file(output_stream)
                output_stream.seek(0)
                self.assertEqual(output_stream.read(), b'12345')

    def test_download_raises_retriable_error_with_truncated_stream(self):
        test_case_headers = [[('Content-Length', '5')], [('Content-Range', 'bytes 15-19/100')]]
        for headers in test_case_headers:
            with self.subTest(headers=headers):
                head_object_response = self.create_response(status_code=200, header=headers)
                media_response = self.create_response(status_code=200, header=headers)
                media_response.read.side_effect = [b'1234', b'']
                self.https_connection.getresponse.side_effect = [head_object_response, media_response]
                bucket = Bucket(self.service_connection, 'bucket')
                key = bucket.get_key('object')
                with self.assertRaisesRegex(ResumableDownloadException, 'Download stream truncated. Received 4 of 5 bytes.') as context:
                    key.get_file(io.BytesIO())
                    self.assertEqual(context.exception.disposition, ResumableTransferDisposition.WAIT_BEFORE_RETRY)