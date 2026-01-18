import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
class UploadTest(unittest.TestCase):

    def setUp(self):
        self.sample_data = b'abc' * 200
        self.sample_stream = six.BytesIO(self.sample_data)
        self.url_builder = base_api._UrlBuilder('http://www.uploads.com')
        self.request = http_wrapper.Request('http://www.uploads.com', headers={'content-type': 'text/plain'})
        self.response = http_wrapper.Response(info={'status': http_client.OK, 'location': 'http://www.uploads.com'}, content='', request_url='http://www.uploads.com')
        self.fail_response = http_wrapper.Response(info={'status': http_client.SERVICE_UNAVAILABLE, 'location': 'http://www.uploads.com'}, content='', request_url='http://www.uploads.com')

    def testStreamInChunksCompressed(self):
        """Test that StreamInChunks will handle compression correctly."""
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
        upload.strategy = transfer.RESUMABLE_UPLOAD
        upload.chunksize = len(self.sample_data)
        with mock.patch.object(transfer.Upload, '_Upload__SendMediaRequest') as mock_result, mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
            mock_result.return_value = self.response
            make_request.return_value = self.response
            upload.InitializeUpload(self.request, 'http')
            upload.StreamInChunks()
            (request, _), _ = mock_result.call_args_list[0]
            self.assertTrue(mock_result.called)
            self.assertEqual(request.headers['Content-Encoding'], 'gzip')
            self.assertLess(len(request.body), len(self.sample_data))

    def testStreamMediaCompressedFail(self):
        """Test that non-chunked uploads raise an exception.

        Ensure uploads with the compressed and resumable flags set called from
        StreamMedia raise an exception. Those uploads are unsupported.
        """
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, auto_transfer=True, gzip_encoded=True)
        upload.strategy = transfer.RESUMABLE_UPLOAD
        with mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
            make_request.return_value = self.response
            upload.InitializeUpload(self.request, 'http')
            with self.assertRaises(exceptions.InvalidUserInputError):
                upload.StreamMedia()

    def testAutoTransferCompressed(self):
        """Test that automatic transfers are compressed.

        Ensure uploads with the compressed, resumable, and automatic transfer
        flags set call StreamInChunks. StreamInChunks is tested in an earlier
        test.
        """
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
        upload.strategy = transfer.RESUMABLE_UPLOAD
        with mock.patch.object(transfer.Upload, 'StreamInChunks') as mock_result, mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
            mock_result.return_value = self.response
            make_request.return_value = self.response
            upload.InitializeUpload(self.request, 'http')
            self.assertTrue(mock_result.called)

    def testMultipartCompressed(self):
        """Test that multipart uploads are compressed."""
        upload_config = base_api.ApiUploadInfo(accept=['*/*'], max_size=None, simple_multipart=True, simple_path=u'/upload')
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
        self.request.body = '{"body_field_one": 7}'
        upload.ConfigureRequest(upload_config, self.request, self.url_builder)
        self.assertEqual(self.url_builder.query_params['uploadType'], 'multipart')
        self.assertEqual(self.request.headers['Content-Encoding'], 'gzip')
        self.assertLess(len(self.request.body), len(self.sample_data))
        with gzip.GzipFile(fileobj=six.BytesIO(self.request.body)) as f:
            original = f.read()
            self.assertTrue(self.sample_data in original)

    def testMediaCompressed(self):
        """Test that media uploads are compressed."""
        upload_config = base_api.ApiUploadInfo(accept=['*/*'], max_size=None, simple_multipart=True, simple_path=u'/upload')
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
        upload.ConfigureRequest(upload_config, self.request, self.url_builder)
        self.assertEqual(self.url_builder.query_params['uploadType'], 'media')
        self.assertEqual(self.request.headers['Content-Encoding'], 'gzip')
        self.assertLess(len(self.request.body), len(self.sample_data))
        with gzip.GzipFile(fileobj=six.BytesIO(self.request.body)) as f:
            original = f.read()
            self.assertTrue(self.sample_data in original)

    def HttpRequestSideEffect(self, responses=None):
        responses = [(response.info, response.content) for response in responses]

        def _side_effect(uri, **kwargs):
            body = kwargs['body']
            read_func = getattr(body, 'read', None)
            if read_func:
                body = read_func()
            self.assertEqual(int(kwargs['headers']['content-length']), len(body))
            return responses.pop(0)
        return _side_effect

    def testRetryRequestChunks(self):
        """Test that StreamInChunks will retry correctly."""
        refresh_response = http_wrapper.Response(info={'status': http_wrapper.RESUME_INCOMPLETE, 'location': 'http://www.uploads.com'}, content='', request_url='http://www.uploads.com')
        bytes_http = httplib2.Http()
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, http=bytes_http)
        upload.strategy = transfer.RESUMABLE_UPLOAD
        upload.chunksize = len(self.sample_data)
        with mock.patch.object(bytes_http, 'request') as make_request:
            responses = [self.response, self.fail_response, refresh_response, self.response]
            make_request.side_effect = self.HttpRequestSideEffect(responses)
            upload.InitializeUpload(self.request, bytes_http)
            upload.StreamInChunks()
            self.assertEquals(make_request.call_count, len(responses))

    def testStreamInChunks(self):
        """Test StreamInChunks."""
        resume_incomplete_responses = [http_wrapper.Response(info={'status': http_wrapper.RESUME_INCOMPLETE, 'location': 'http://www.uploads.com', 'range': '0-{}'.format(end)}, content='', request_url='http://www.uploads.com') for end in [199, 399, 599]]
        responses = [self.response] + resume_incomplete_responses + [self.response]
        bytes_http = httplib2.Http()
        upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, http=bytes_http)
        upload.strategy = transfer.RESUMABLE_UPLOAD
        upload.chunksize = 200
        with mock.patch.object(bytes_http, 'request') as make_request:
            make_request.side_effect = self.HttpRequestSideEffect(responses)
            upload.InitializeUpload(self.request, bytes_http)
            upload.StreamInChunks()
            self.assertEquals(make_request.call_count, len(responses))

    @mock.patch.object(transfer.Upload, 'RefreshResumableUploadState', new=mock.Mock())
    def testFinalizesTransferUrlIfClientPresent(self):
        """Tests upload's enforcement of client custom endpoints."""
        mock_client = mock.Mock()
        mock_http = mock.Mock()
        fake_json_data = json.dumps({'auto_transfer': False, 'mime_type': '', 'total_size': 0, 'url': 'url'})
        transfer.Upload.FromData(self.sample_stream, fake_json_data, mock_http, client=mock_client)
        mock_client.FinalizeTransferUrl.assert_called_once_with('url')