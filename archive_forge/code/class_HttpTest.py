import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
class HttpTest(unittest.TestCase):

    def testReadHeader(self):
        status, header, status_message = read_headers(HeaderSockMock('data/header01.txt'))
        self.assertEqual(status, 101)
        self.assertEqual(header['connection'], 'Upgrade')
        self.assertRaises(ws.WebSocketException, read_headers, HeaderSockMock('data/header02.txt'))

    def testTunnel(self):
        self.assertRaises(ws.WebSocketProxyException, _tunnel, HeaderSockMock('data/header01.txt'), 'example.com', 80, ('username', 'password'))
        self.assertRaises(ws.WebSocketProxyException, _tunnel, HeaderSockMock('data/header02.txt'), 'example.com', 80, ('username', 'password'))

    @unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
    def testConnect(self):
        if ws._http.HAVE_PYTHON_SOCKS:
            self.assertRaises((ProxyTimeoutError, OSError), _start_proxied_socket, 'wss://example.com', OptsList(), proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='socks4', http_proxy_timeout=1))
            self.assertRaises((ProxyTimeoutError, OSError), _start_proxied_socket, 'wss://example.com', OptsList(), proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='socks4a', http_proxy_timeout=1))
            self.assertRaises((ProxyTimeoutError, OSError), _start_proxied_socket, 'wss://example.com', OptsList(), proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='socks5', http_proxy_timeout=1))
            self.assertRaises((ProxyTimeoutError, OSError), _start_proxied_socket, 'wss://example.com', OptsList(), proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='socks5h', http_proxy_timeout=1))
            self.assertRaises(ProxyConnectionError, connect, 'wss://example.com', OptsList(), proxy_info(http_proxy_host='127.0.0.1', http_proxy_port=9999, proxy_type='socks4', http_proxy_timeout=1), None)
        self.assertRaises(TypeError, _get_addrinfo_list, None, 80, True, proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='9999', proxy_type='http'))
        self.assertRaises(TypeError, _get_addrinfo_list, None, 80, True, proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='9999', proxy_type='http'))
        self.assertRaises(socket.timeout, connect, 'wss://google.com', OptsList(), proxy_info(http_proxy_host='8.8.8.8', http_proxy_port=9999, proxy_type='http', http_proxy_timeout=1), None)
        self.assertEqual(connect('wss://google.com', OptsList(), proxy_info(http_proxy_host='8.8.8.8', http_proxy_port=8080, proxy_type='http'), True), (True, ('google.com', 443, '/')))

    @unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
    @unittest.skipUnless(TEST_WITH_PROXY, 'This test requires a HTTP proxy to be running on port 8899')
    @unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
    def testProxyConnect(self):
        ws = websocket.WebSocket()
        ws.connect(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', http_proxy_host='127.0.0.1', http_proxy_port='8899', proxy_type='http')
        ws.send('Hello, Server')
        server_response = ws.recv()
        self.assertEqual(server_response, 'Hello, Server')
        self.assertEqual(_get_addrinfo_list('api.bitfinex.com', 443, True, proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8899', proxy_type='http')), (socket.getaddrinfo('127.0.0.1', 8899, 0, socket.SOCK_STREAM, socket.SOL_TCP), True, None))
        self.assertEqual(connect('wss://api.bitfinex.com/ws/2', OptsList(), proxy_info(http_proxy_host='127.0.0.1', http_proxy_port=8899, proxy_type='http'), None)[1], ('api.bitfinex.com', 443, '/ws/2'))

    @unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
    def testSSLopt(self):
        ssloptions = {'check_hostname': False, 'server_hostname': 'ServerName', 'ssl_version': ssl.PROTOCOL_TLS_CLIENT, 'ciphers': 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:                        TLS_AES_128_GCM_SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:                        ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:                        ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:                        DHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:                        ECDHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA256:                        ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA384:                        DHE-RSA-AES256-SHA256:ECDHE-ECDSA-AES128-SHA256:                        ECDHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA256:                        ECDHE-ECDSA-AES256-SHA:ECDHE-RSA-AES256-SHA', 'ecdh_curve': 'prime256v1'}
        ws_ssl1 = websocket.WebSocket(sslopt=ssloptions)
        ws_ssl1.connect('wss://api.bitfinex.com/ws/2')
        ws_ssl1.send('Hello')
        ws_ssl1.close()
        ws_ssl2 = websocket.WebSocket(sslopt={'check_hostname': True})
        ws_ssl2.connect('wss://api.bitfinex.com/ws/2')
        ws_ssl2.close

    def testProxyInfo(self):
        self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').proxy_protocol, 'http')
        self.assertRaises(ProxyError, proxy_info, http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='badval')
        self.assertEqual(proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='http').proxy_host, 'example.com')
        self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').proxy_port, '8080')
        self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').auth, None)
        self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http', http_proxy_auth=('my_username123', 'my_pass321')).auth[0], 'my_username123')
        self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http', http_proxy_auth=('my_username123', 'my_pass321')).auth[1], 'my_pass321')