import os
import os.path
import socket
import unittest
from base64 import decodebytes as base64decode
import websocket as ws
from websocket._handshake import _create_sec_websocket_key
from websocket._handshake import _validate as _validate_header
from websocket._http import read_headers
from websocket._utils import validate_utf8
class HandshakeTest(unittest.TestCase):

    @unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
    def test_http_SSL(self):
        websock1 = ws.WebSocket(sslopt={'cert_chain': ssl.get_default_verify_paths().capath}, enable_multithread=False)
        self.assertRaises(ValueError, websock1.connect, 'wss://api.bitfinex.com/ws/2')
        websock2 = ws.WebSocket(sslopt={'certfile': 'myNonexistentCertFile'})
        self.assertRaises(FileNotFoundError, websock2.connect, 'wss://api.bitfinex.com/ws/2')

    @unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
    def testManualHeaders(self):
        websock3 = ws.WebSocket(sslopt={'ca_certs': ssl.get_default_verify_paths().cafile, 'ca_cert_path': ssl.get_default_verify_paths().capath})
        self.assertRaises(ws._exceptions.WebSocketBadStatusException, websock3.connect, 'wss://api.bitfinex.com/ws/2', cookie='chocolate', origin='testing_websockets.com', host='echo.websocket.events/websocket-client-test', subprotocols=['testproto'], connection='Upgrade', header={'CustomHeader1': '123', 'Cookie': 'TestValue', 'Sec-WebSocket-Key': 'k9kFAUWNAMmf5OEMfTlOEA==', 'Sec-WebSocket-Protocol': 'newprotocol'})

    def testIPv6(self):
        websock2 = ws.WebSocket()
        self.assertRaises(ValueError, websock2.connect, '2001:4860:4860::8888')

    def testBadURLs(self):
        websock3 = ws.WebSocket()
        self.assertRaises(ValueError, websock3.connect, 'ws//example.com')
        self.assertRaises(ws.WebSocketAddressException, websock3.connect, 'ws://example')
        self.assertRaises(ValueError, websock3.connect, 'example.com')