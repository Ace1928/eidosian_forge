from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import email
from gslib.gcs_json_media import BytesTransferredContainer
from gslib.gcs_json_media import HttpWithDownloadStream
from gslib.gcs_json_media import UploadCallbackConnectionClassFactory
import gslib.tests.testcase as testcase
import httplib2
import io
import six
from six import add_move, MovedModule
from six.moves import http_client
from six.moves import mock
class TestUploadCallbackConnection(testcase.GsUtilUnitTestCase):
    """Tests for the upload callback connection."""

    def setUp(self):
        super(TestUploadCallbackConnection, self).setUp()
        self.bytes_container = BytesTransferredContainer()
        self.class_factory = UploadCallbackConnectionClassFactory(self.bytes_container, buffer_size=50, total_size=100, progress_callback='Sample')
        self.instance = self.class_factory.GetConnectionClass()('host')

    @mock.patch(https_connection)
    def testHeaderDefaultBehavior(self, mock_conn):
        """Test the size modifier is correct under expected headers."""
        mock_conn.putheader.return_value = None
        self.instance.putheader('content-encoding', 'gzip')
        self.instance.putheader('content-length', '10')
        self.instance.putheader('content-range', 'bytes 0-104/*')
        self.assertAlmostEqual(self.instance.size_modifier, 10.5)

    @mock.patch(https_connection)
    def testHeaderIgnoreWithoutGzip(self, mock_conn):
        """Test that the gzip content-encoding is required to modify size."""
        mock_conn.putheader.return_value = None
        self.instance.putheader('content-length', '10')
        self.instance.putheader('content-range', 'bytes 0-99/*')
        self.assertAlmostEqual(self.instance.size_modifier, 1.0)

    @mock.patch(https_connection)
    def testHeaderRangeFormatX_YSlashStar(self, mock_conn):
        """Test content-range header format X-Y/* """
        self.instance.putheader('content-encoding', 'gzip')
        self.instance.putheader('content-length', '10')
        self.instance.putheader('content-range', 'bytes 0-99/*')
        self.assertAlmostEqual(self.instance.size_modifier, 10.0)

    @mock.patch(https_connection)
    def testHeaderRangeFormatStarSlash100(self, mock_conn):
        """Test content-range header format */100 """
        self.instance.putheader('content-encoding', 'gzip')
        self.instance.putheader('content-length', '10')
        self.instance.putheader('content-range', 'bytes */100')
        self.assertAlmostEqual(self.instance.size_modifier, 1.0)

    @mock.patch(https_connection)
    def testHeaderRangeFormat0_99Slash100(self, mock_conn):
        """Test content-range header format 0-99/100 """
        self.instance.putheader('content-encoding', 'gzip')
        self.instance.putheader('content-length', '10')
        self.instance.putheader('content-range', 'bytes 0-99/100')
        self.assertAlmostEqual(self.instance.size_modifier, 10.0)

    @mock.patch(https_connection)
    def testHeaderParseFailure(self, mock_conn):
        """Test incorrect header values do not raise exceptions."""
        mock_conn.putheader.return_value = None
        self.instance.putheader('content-encoding', 'gzip')
        self.instance.putheader('content-length', 'bytes 10')
        self.instance.putheader('content-range', 'not a number')
        self.assertAlmostEqual(self.instance.size_modifier, 1.0)

    @mock.patch('gslib.progress_callback.ProgressCallbackWithTimeout')
    @mock.patch('httplib2.HTTPSConnectionWithTimeout')
    def testSendDefaultBehavior(self, mock_conn, mock_callback):
        mock_conn.send.return_value = None
        self.instance.size_modifier = 2
        self.instance.processed_initial_bytes = True
        self.instance.callback_processor = mock_callback
        sample_data = b'0123456789'
        self.instance.send(sample_data)
        self.assertTrue(mock_conn.send.called)
        (_, sent_data), _ = mock_conn.send.call_args_list[0]
        self.assertEqual(sent_data, sample_data)
        self.assertTrue(mock_callback.Progress.called)
        [sent_bytes], _ = mock_callback.Progress.call_args_list[0]
        self.assertEqual(sent_bytes, 20)