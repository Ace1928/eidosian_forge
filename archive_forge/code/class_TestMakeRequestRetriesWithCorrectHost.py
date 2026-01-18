from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
class TestMakeRequestRetriesWithCorrectHost(AWSMockServiceTestCase):

    def setUp(self):
        self.connection = AWSAuthConnection('s3.amazonaws.com')
        self.non_retriable_code = 404
        self.retry_status_codes = [301, 400]
        self.success_response = self.create_response(200)
        self.default_host = 'bucket.s3.amazonaws.com'
        self.retry_region = RETRY_REGION_BYTES.decode('utf-8')
        self.default_retried_host = 'bucket.s3.%s.amazonaws.com' % self.retry_region
        self.test_headers = [('x-amz-bucket-region', self.retry_region)]

    def test_non_retriable_status_returns_original_response(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            error_response = self.create_response(self.non_retriable_code)
            mocked_mexe.side_effect = [error_response]
            response = self.connection.make_request('HEAD', '/', host=self.default_host)
            self.assertEqual(response, error_response)
            self.assertEqual(mocked_mexe.call_count, 1)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_host)

    def test_non_retriable_host_returns_original_response(self):
        for code in self.retry_status_codes:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                error_response = self.create_response(code)
                mocked_mexe.side_effect = [error_response]
                other_host = 'bucket.some-other-provider.com'
                response = self.connection.make_request('HEAD', '/', host=other_host)
                self.assertEqual(response, error_response)
                self.assertEqual(mocked_mexe.call_count, 1)
                self.assertEqual(mocked_mexe.call_args[0][0].host, other_host)

    def test_non_retriable_status_raises_original_exception(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            error_response = S3ResponseError(self.non_retriable_code, 'reason')
            mocked_mexe.side_effect = [error_response]
            with self.assertRaises(S3ResponseError) as cm:
                self.connection.make_request('HEAD', '/', host=self.default_host)
            self.assertEqual(cm.exception, error_response)
            self.assertEqual(mocked_mexe.call_count, 1)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_host)

    def test_non_retriable_host_raises_original_exception(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            error_response = S3ResponseError(self.non_retriable_code, 'reason')
            mocked_mexe.side_effect = [error_response]
            other_host = 'bucket.some-other-provider.com'
            with self.assertRaises(S3ResponseError) as cm:
                self.connection.make_request('HEAD', '/', host=other_host)
            self.assertEqual(cm.exception, error_response)
            self.assertEqual(mocked_mexe.call_count, 1)
            self.assertEqual(mocked_mexe.call_args[0][0].host, other_host)

    def test_response_retries_from_callable_headers(self):
        for code in self.retry_status_codes:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [self.create_response(code, header=self.test_headers), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 2)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_retry_changes_host_with_region(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            mocked_mexe.side_effect = [self.create_response(400, header=self.test_headers), self.success_response]
            response = self.connection.make_request('HEAD', '/', host=self.default_host)
            self.assertEqual(response, self.success_response)
            self.assertEqual(mocked_mexe.call_count, 2)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_retry_changes_host_with_multiple_s3_occurrences(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            mocked_mexe.side_effect = [self.create_response(400, header=self.test_headers), self.success_response]
            response = self.connection.make_request('HEAD', '/', host='a.s3.a.s3.amazonaws.com')
            self.assertEqual(response, self.success_response)
            self.assertEqual(mocked_mexe.call_count, 2)
            self.assertEqual(mocked_mexe.call_args[0][0].host, 'a.s3.a.s3.us-east-2.amazonaws.com')

    def test_retry_changes_host_with_s3_in_region(self):
        with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
            mocked_mexe.side_effect = [self.create_response(400, header=self.test_headers), self.success_response]
            response = self.connection.make_request('HEAD', '/', host='bucket.s3.asdf-s3.amazonaws.com')
            self.assertEqual(response, self.success_response)
            self.assertEqual(mocked_mexe.call_count, 2)
            self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_response_body_parsed_for_region(self):
        for code, body in ERRORS_WITH_REGION_IN_BODY:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [self.create_response(code, body=body), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 2)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_error_body_parsed_for_region(self):
        for code, body in ERRORS_WITH_REGION_IN_BODY:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [S3ResponseError(code, 'reason', body=body), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 2)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_response_without_region_header_retries_from_bucket_head(self):
        for code in self.retry_status_codes:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [self.create_response(code), self.create_response(200, header=self.test_headers), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 3)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_response_body_without_region_sends_bucket_head(self):
        for code, body in ERRORS_WITHOUT_REGION_IN_BODY:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [self.create_response(code, body=body), self.create_response(200, header=self.test_headers), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 3)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_error_body_without_region_retries_from_bucket_head_request(self):
        for code, body in ERRORS_WITHOUT_REGION_IN_BODY:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                mocked_mexe.side_effect = [S3ResponseError(code, 'reason', body=body), self.create_response(200, header=self.test_headers), self.success_response]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, self.success_response)
                self.assertEqual(mocked_mexe.call_count, 3)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_retried_host)

    def test_retry_head_request_lacks_region_returns_original_response(self):
        for code in self.retry_status_codes:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                error_response = self.create_response(code)
                mocked_mexe.side_effect = [error_response, self.create_response(200, header=[])]
                response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(response, error_response)
                self.assertEqual(mocked_mexe.call_count, 2)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_host)

    def test_retry_head_request_lacks_region_raises_original_exception(self):
        for code in self.retry_status_codes:
            with mock.patch.object(self.connection, '_mexe') as mocked_mexe:
                error_response = S3ResponseError(code, 'reason')
                mocked_mexe.side_effect = [error_response, self.create_response(200, header=[])]
                with self.assertRaises(S3ResponseError) as cm:
                    response = self.connection.make_request('HEAD', '/', host=self.default_host)
                self.assertEqual(cm.exception, error_response)
                self.assertEqual(mocked_mexe.call_count, 2)
                self.assertEqual(mocked_mexe.call_args[0][0].host, self.default_host)