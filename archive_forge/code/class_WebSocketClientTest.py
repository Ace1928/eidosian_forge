from unittest import mock
import testtools
from zunclient.common.websocketclient import websocketclient
class WebSocketClientTest(testtools.TestCase):

    def test_websocketclient_variables(self):
        mock_client = mock.Mock()
        wsclient = websocketclient.WebSocketClient(zunclient=mock_client, url=URL, id=CONTAINER_ID, escape=ESCAPE_FLAG, close_wait=WAIT_TIME)
        self.assertEqual(wsclient.url, URL)
        self.assertEqual(wsclient.id, CONTAINER_ID)
        self.assertEqual(wsclient.escape, ESCAPE_FLAG)
        self.assertEqual(wsclient.close_wait, WAIT_TIME)