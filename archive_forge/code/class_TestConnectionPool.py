import io
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
from test import LONG_TIMEOUT, SHORT_TIMEOUT, onlyPy2
from threading import Event
import mock
import pytest
import six
from dummyserver.server import HAS_IPV6_AND_DNS, NoIPv6Warning
from dummyserver.testcase import HTTPDummyServerTestCase, SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, encode_multipart_formdata
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six import b, u
from urllib3.packages.six.moves.urllib.parse import urlencode
from urllib3.util import SKIP_HEADER, SKIPPABLE_HEADERS
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout
from .. import INVALID_SOURCE_ADDRESSES, TARPIT_HOST, VALID_SOURCE_ADDRESSES
from ..port_helpers import find_unused_port
class TestConnectionPool(HTTPDummyServerTestCase):

    def test_get(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/specific_method', fields={'method': 'GET'})
            assert r.status == 200, r.data

    def test_post_url(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('POST', '/specific_method', fields={'method': 'POST'})
            assert r.status == 200, r.data

    def test_urlopen_put(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.urlopen('PUT', '/specific_method?method=PUT')
            assert r.status == 200, r.data

    def test_wrong_specific_method(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/specific_method', fields={'method': 'POST'})
            assert r.status == 400, r.data
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('POST', '/specific_method', fields={'method': 'GET'})
            assert r.status == 400, r.data

    def test_upload(self):
        data = "I'm in ur multipart form-data, hazing a cheezburgr"
        fields = {'upload_param': 'filefield', 'upload_filename': 'lolcat.txt', 'upload_size': len(data), 'filefield': ('lolcat.txt', data)}
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('POST', '/upload', fields=fields)
            assert r.status == 200, r.data

    def test_one_name_multiple_values(self):
        fields = [('foo', 'a'), ('foo', 'b')]
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/echo', fields=fields)
            assert r.data == b'foo=a&foo=b'
            r = pool.request('POST', '/echo', fields=fields)
            assert r.data.count(b'name="foo"') == 2

    def test_request_method_body(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            body = b'hi'
            r = pool.request('POST', '/echo', body=body)
            assert r.data == body
            fields = [('hi', 'hello')]
            with pytest.raises(TypeError):
                pool.request('POST', '/echo', body=body, fields=fields)

    def test_unicode_upload(self):
        fieldname = u('myfile')
        filename = u('â\x99¥.txt')
        data = u('â\x99¥').encode('utf8')
        size = len(data)
        fields = {u('upload_param'): fieldname, u('upload_filename'): filename, u('upload_size'): size, fieldname: (filename, data)}
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('POST', '/upload', fields=fields)
            assert r.status == 200, r.data

    def test_nagle(self):
        """Test that connections have TCP_NODELAY turned on"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            conn = pool._get_conn()
            try:
                pool._make_request(conn, 'GET', '/')
                tcp_nodelay_setting = conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                assert tcp_nodelay_setting
            finally:
                conn.close()

    def test_socket_options(self):
        """Test that connections accept socket options."""
        with HTTPConnectionPool(self.host, self.port, socket_options=[(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]) as pool:
            s = pool._new_conn()._new_conn()
            try:
                using_keepalive = s.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) > 0
                assert using_keepalive
            finally:
                s.close()

    def test_disable_default_socket_options(self):
        """Test that passing None disables all socket options."""
        with HTTPConnectionPool(self.host, self.port, socket_options=None) as pool:
            s = pool._new_conn()._new_conn()
            try:
                using_nagle = s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 0
                assert using_nagle
            finally:
                s.close()

    def test_defaults_are_applied(self):
        """Test that modifying the default socket options works."""
        with HTTPConnectionPool(self.host, self.port) as pool:
            conn = pool._new_conn()
            try:
                conn.default_socket_options += [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
                s = conn._new_conn()
                nagle_disabled = s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) > 0
                using_keepalive = s.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) > 0
                assert nagle_disabled
                assert using_keepalive
            finally:
                conn.close()
                s.close()

    def test_connection_error_retries(self):
        """ECONNREFUSED error should raise a connection error, with retries"""
        port = find_unused_port()
        with HTTPConnectionPool(self.host, port) as pool:
            with pytest.raises(MaxRetryError) as e:
                pool.request('GET', '/', retries=Retry(connect=3))
            assert type(e.value.reason) == NewConnectionError

    def test_timeout_success(self):
        timeout = Timeout(connect=3, read=5, total=None)
        with HTTPConnectionPool(self.host, self.port, timeout=timeout) as pool:
            pool.request('GET', '/')
            pool.request('GET', '/')
        with HTTPConnectionPool(self.host, self.port, timeout=timeout) as pool:
            pool.request('GET', '/')
        timeout = Timeout(total=None)
        with HTTPConnectionPool(self.host, self.port, timeout=timeout) as pool:
            pool.request('GET', '/')

    def test_tunnel(self):
        timeout = Timeout(total=None)
        with HTTPConnectionPool(self.host, self.port, timeout=timeout) as pool:
            conn = pool._get_conn()
            try:
                conn.set_tunnel(self.host, self.port)
                conn._tunnel = mock.Mock(return_value=None)
                pool._make_request(conn, 'GET', '/')
                conn._tunnel.assert_called_once_with()
            finally:
                conn.close()
        timeout = Timeout(total=None)
        with HTTPConnectionPool(self.host, self.port, timeout=timeout) as pool:
            conn = pool._get_conn()
            try:
                conn._tunnel = mock.Mock(return_value=None)
                pool._make_request(conn, 'GET', '/')
                assert not conn._tunnel.called
            finally:
                conn.close()

    def test_redirect(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/redirect', fields={'target': '/'}, redirect=False)
            assert r.status == 303
            r = pool.request('GET', '/redirect', fields={'target': '/'})
            assert r.status == 200
            assert r.data == b'Dummy server!'

    def test_bad_connect(self):
        with HTTPConnectionPool('badhost.invalid', self.port) as pool:
            with pytest.raises(MaxRetryError) as e:
                pool.request('GET', '/', retries=5)
            assert type(e.value.reason) == NewConnectionError

    def test_keepalive(self):
        with HTTPConnectionPool(self.host, self.port, block=True, maxsize=1) as pool:
            r = pool.request('GET', '/keepalive?close=0')
            r = pool.request('GET', '/keepalive?close=0')
            assert r.status == 200
            assert pool.num_connections == 1
            assert pool.num_requests == 2

    def test_keepalive_close(self):
        with HTTPConnectionPool(self.host, self.port, block=True, maxsize=1, timeout=2) as pool:
            r = pool.request('GET', '/keepalive?close=1', retries=0, headers={'Connection': 'close'})
            assert pool.num_connections == 1
            conn = pool.pool.get()
            assert conn.sock is None
            pool._put_conn(conn)
            r = pool.request('GET', '/keepalive?close=0', retries=0, headers={'Connection': 'keep-alive'})
            conn = pool.pool.get()
            assert conn.sock is not None
            pool._put_conn(conn)
            r = pool.request('GET', '/keepalive?close=1', retries=0, headers={'Connection': 'close'})
            assert r.status == 200
            conn = pool.pool.get()
            assert conn.sock is None
            pool._put_conn(conn)
            r = pool.request('GET', '/keepalive?close=0')

    def test_post_with_urlencode(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            data = {'banana': 'hammock', 'lol': 'cat'}
            r = pool.request('POST', '/echo', fields=data, encode_multipart=False)
            assert r.data.decode('utf-8') == urlencode(data)

    def test_post_with_multipart(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            data = {'banana': 'hammock', 'lol': 'cat'}
            r = pool.request('POST', '/echo', fields=data, encode_multipart=True)
            body = r.data.split(b'\r\n')
            encoded_data = encode_multipart_formdata(data)[0]
            expected_body = encoded_data.split(b'\r\n')
            '\n            We need to loop the return lines because a timestamp is attached\n            from within encode_multipart_formdata. When the server echos back\n            the data, it has the timestamp from when the data was encoded, which\n            is not equivalent to when we run encode_multipart_formdata on\n            the data again.\n            '
            for i, line in enumerate(body):
                if line.startswith(b'--'):
                    continue
                assert body[i] == expected_body[i]

    def test_post_with_multipart__iter__(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            data = {'hello': 'world'}
            r = pool.request('POST', '/echo', fields=data, preload_content=False, multipart_boundary='boundary', encode_multipart=True)
            chunks = [chunk for chunk in r]
            assert chunks == [b'--boundary\r\n', b'Content-Disposition: form-data; name="hello"\r\n', b'\r\n', b'world\r\n', b'--boundary--\r\n']

    def test_check_gzip(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/encodingrequest', headers={'accept-encoding': 'gzip'})
            assert r.headers.get('content-encoding') == 'gzip'
            assert r.data == b'hello, world!'

    def test_check_deflate(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/encodingrequest', headers={'accept-encoding': 'deflate'})
            assert r.headers.get('content-encoding') == 'deflate'
            assert r.data == b'hello, world!'

    def test_bad_decode(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            with pytest.raises(DecodeError):
                pool.request('GET', '/encodingrequest', headers={'accept-encoding': 'garbage-deflate'})
            with pytest.raises(DecodeError):
                pool.request('GET', '/encodingrequest', headers={'accept-encoding': 'garbage-gzip'})

    def test_connection_count(self):
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as pool:
            pool.request('GET', '/')
            pool.request('GET', '/')
            pool.request('GET', '/')
            assert pool.num_connections == 1
            assert pool.num_requests == 3

    def test_connection_count_bigpool(self):
        with HTTPConnectionPool(self.host, self.port, maxsize=16) as http_pool:
            http_pool.request('GET', '/')
            http_pool.request('GET', '/')
            http_pool.request('GET', '/')
            assert http_pool.num_connections == 1
            assert http_pool.num_requests == 3

    def test_partial_response(self):
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as pool:
            req_data = {'lol': 'cat'}
            resp_data = urlencode(req_data).encode('utf-8')
            r = pool.request('GET', '/echo', fields=req_data, preload_content=False)
            assert r.read(5) == resp_data[:5]
            assert r.read() == resp_data[5:]

    def test_lazy_load_twice(self):
        with HTTPConnectionPool(self.host, self.port, block=True, maxsize=1, timeout=2) as pool:
            payload_size = 1024 * 2
            first_chunk = 512
            boundary = 'foo'
            req_data = {'count': 'a' * payload_size}
            resp_data = encode_multipart_formdata(req_data, boundary=boundary)[0]
            req2_data = {'count': 'b' * payload_size}
            resp2_data = encode_multipart_formdata(req2_data, boundary=boundary)[0]
            r1 = pool.request('POST', '/echo', fields=req_data, multipart_boundary=boundary, preload_content=False)
            assert r1.read(first_chunk) == resp_data[:first_chunk]
            try:
                r2 = pool.request('POST', '/echo', fields=req2_data, multipart_boundary=boundary, preload_content=False, pool_timeout=0.001)
                assert r2.read(first_chunk) == resp2_data[:first_chunk]
                assert r1.read() == resp_data[first_chunk:]
                assert r2.read() == resp2_data[first_chunk:]
                assert pool.num_requests == 2
            except EmptyPoolError:
                assert r1.read() == resp_data[first_chunk:]
                assert pool.num_requests == 1
            assert pool.num_connections == 1

    def test_for_double_release(self):
        MAXSIZE = 5
        with HTTPConnectionPool(self.host, self.port, maxsize=MAXSIZE) as pool:
            assert pool.num_connections == 0
            assert pool.pool.qsize() == MAXSIZE
            pool.pool.get()
            assert pool.pool.qsize() == MAXSIZE - 1
            pool.urlopen('GET', '/')
            assert pool.pool.qsize() == MAXSIZE - 1
            pool.urlopen('GET', '/', preload_content=False)
            assert pool.pool.qsize() == MAXSIZE - 2
            pool.urlopen('GET', '/')
            assert pool.pool.qsize() == MAXSIZE - 2
            pool.urlopen('GET', '/').data
            assert pool.pool.qsize() == MAXSIZE - 2
            pool.urlopen('GET', '/')
            assert pool.pool.qsize() == MAXSIZE - 2

    def test_release_conn_parameter(self):
        MAXSIZE = 5
        with HTTPConnectionPool(self.host, self.port, maxsize=MAXSIZE) as pool:
            assert pool.pool.qsize() == MAXSIZE
            pool.request('GET', '/', release_conn=False, preload_content=False)
            assert pool.pool.qsize() == MAXSIZE - 1

    def test_dns_error(self):
        with HTTPConnectionPool('thishostdoesnotexist.invalid', self.port, timeout=0.001) as pool:
            with pytest.raises(MaxRetryError):
                pool.request('GET', '/test', retries=2)

    @pytest.mark.parametrize('char', [' ', '\r', '\n', '\x00'])
    def test_invalid_method_not_allowed(self, char):
        with pytest.raises(ValueError):
            with HTTPConnectionPool(self.host, self.port) as pool:
                pool.request('GET' + char, '/')

    def test_percent_encode_invalid_target_chars(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/echo_params?q=\r&k=\n \n')
            assert r.data == b"[('k', '\\n \\n'), ('q', '\\r')]"

    @pytest.mark.skipif(six.PY2 and platform.system() == 'Darwin' and (os.environ.get('GITHUB_ACTIONS') == 'true'), reason='fails on macOS 2.7 in GitHub Actions for an unknown reason')
    def test_source_address(self):
        for addr, is_ipv6 in VALID_SOURCE_ADDRESSES:
            if is_ipv6 and (not HAS_IPV6_AND_DNS):
                warnings.warn('No IPv6 support: skipping.', NoIPv6Warning)
                continue
            with HTTPConnectionPool(self.host, self.port, source_address=addr, retries=False) as pool:
                r = pool.request('GET', '/source_address')
                assert r.data == b(addr[0])

    @pytest.mark.skipif(six.PY2 and platform.system() == 'Darwin' and (os.environ.get('GITHUB_ACTIONS') == 'true'), reason='fails on macOS 2.7 in GitHub Actions for an unknown reason')
    def test_source_address_error(self):
        for addr in INVALID_SOURCE_ADDRESSES:
            with HTTPConnectionPool(self.host, self.port, source_address=addr, retries=False) as pool:
                with pytest.raises(NewConnectionError):
                    pool.request('GET', '/source_address?{0}'.format(addr))

    def test_stream_keepalive(self):
        x = 2
        with HTTPConnectionPool(self.host, self.port) as pool:
            for _ in range(x):
                response = pool.request('GET', '/chunked', headers={'Connection': 'keep-alive'}, preload_content=False, retries=False)
                for chunk in response.stream():
                    assert chunk == b'123'
            assert pool.num_connections == 1
            assert pool.num_requests == x

    def test_read_chunked_short_circuit(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/chunked', preload_content=False)
            response.read()
            with pytest.raises(StopIteration):
                next(response.read_chunked())

    def test_read_chunked_on_closed_response(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/chunked', preload_content=False)
            response.close()
            with pytest.raises(StopIteration):
                next(response.read_chunked())

    def test_chunked_gzip(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/chunked_gzip', preload_content=False, decode_content=True)
            assert b'123' * 4 == response.read()

    def test_cleanup_on_connection_error(self):
        """
        Test that connections are recycled to the pool on
        connection errors where no http response is received.
        """
        poolsize = 3
        with HTTPConnectionPool(self.host, self.port, maxsize=poolsize, block=True) as http:
            assert http.pool.qsize() == poolsize
            with pytest.raises(MaxRetryError):
                http.request('GET', '/redirect', fields={'target': '/'}, release_conn=False, retries=0)
            r = http.request('GET', '/redirect', fields={'target': '/'}, release_conn=False, retries=1)
            r.release_conn()
            assert http.pool.qsize() == http.pool.maxsize

    def test_mixed_case_hostname(self):
        with HTTPConnectionPool('LoCaLhOsT', self.port) as pool:
            response = pool.request('GET', 'http://LoCaLhOsT:%d/' % self.port)
            assert response.status == 200

    def test_preserves_path_dot_segments(self):
        """ConnectionPool preserves dot segments in the URI"""
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/echo_uri/seg0/../seg2')
            assert response.data == b'/echo_uri/seg0/../seg2'

    def test_default_user_agent_header(self):
        """ConnectionPool has a default user agent"""
        default_ua = _get_default_user_agent()
        custom_ua = "I'm not a web scraper, what are you talking about?"
        custom_ua2 = 'Yet Another User Agent'
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/headers')
            request_headers = json.loads(r.data.decode('utf8'))
            assert request_headers.get('User-Agent') == _get_default_user_agent()
            headers = {'UsEr-AGENt': custom_ua}
            r = pool.request('GET', '/headers', headers=headers)
            request_headers = json.loads(r.data.decode('utf8'))
            assert request_headers.get('User-Agent') == custom_ua
            pool_headers = {'foo': 'bar'}
            pool.headers = pool_headers
            r = pool.request('GET', '/headers')
            request_headers = json.loads(r.data.decode('utf8'))
            assert request_headers.get('User-Agent') == default_ua
            assert 'User-Agent' not in pool_headers
            pool.headers.update({'User-Agent': custom_ua2})
            r = pool.request('GET', '/headers')
            request_headers = json.loads(r.data.decode('utf8'))
            assert request_headers.get('User-Agent') == custom_ua2

    @pytest.mark.parametrize('headers', [None, {}, {'User-Agent': 'key'}, {'user-agent': 'key'}, {b'uSeR-AgEnT': b'key'}, {b'user-agent': 'key'}])
    @pytest.mark.parametrize('chunked', [True, False])
    def test_user_agent_header_not_sent_twice(self, headers, chunked):
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/headers', headers=headers, chunked=chunked)
            request_headers = json.loads(r.data.decode('utf8'))
            if not headers:
                assert request_headers['User-Agent'].startswith('python-urllib3/')
                assert 'key' not in request_headers['User-Agent']
            else:
                assert request_headers['User-Agent'] == 'key'

    def test_no_user_agent_header(self):
        """ConnectionPool can suppress sending a user agent header"""
        custom_ua = "I'm not a web scraper, what are you talking about?"
        with HTTPConnectionPool(self.host, self.port) as pool:
            no_ua_headers = {'User-Agent': SKIP_HEADER}
            r = pool.request('GET', '/headers', headers=no_ua_headers)
            request_headers = json.loads(r.data.decode('utf8'))
            assert 'User-Agent' not in request_headers
            assert no_ua_headers['User-Agent'] == SKIP_HEADER
            pool.headers = no_ua_headers
            r = pool.request('GET', '/headers')
            request_headers = json.loads(r.data.decode('utf8'))
            assert 'User-Agent' not in request_headers
            assert no_ua_headers['User-Agent'] == SKIP_HEADER
            pool_headers = {'User-Agent': custom_ua}
            pool.headers = pool_headers
            r = pool.request('GET', '/headers', headers=no_ua_headers)
            request_headers = json.loads(r.data.decode('utf8'))
            assert 'User-Agent' not in request_headers
            assert no_ua_headers['User-Agent'] == SKIP_HEADER
            assert pool_headers.get('User-Agent') == custom_ua

    @pytest.mark.parametrize('accept_encoding', ['Accept-Encoding', 'accept-encoding', b'Accept-Encoding', b'accept-encoding', None])
    @pytest.mark.parametrize('host', ['Host', 'host', b'Host', b'host', None])
    @pytest.mark.parametrize('user_agent', ['User-Agent', 'user-agent', b'User-Agent', b'user-agent', None])
    @pytest.mark.parametrize('chunked', [True, False])
    def test_skip_header(self, accept_encoding, host, user_agent, chunked):
        headers = {}
        if accept_encoding is not None:
            headers[accept_encoding] = SKIP_HEADER
        if host is not None:
            headers[host] = SKIP_HEADER
        if user_agent is not None:
            headers[user_agent] = SKIP_HEADER
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/headers', headers=headers, chunked=chunked)
        request_headers = json.loads(r.data.decode('utf8'))
        if accept_encoding is None:
            assert 'Accept-Encoding' in request_headers
        else:
            assert accept_encoding not in request_headers
        if host is None:
            assert 'Host' in request_headers
        else:
            assert host not in request_headers
        if user_agent is None:
            assert 'User-Agent' in request_headers
        else:
            assert user_agent not in request_headers

    @pytest.mark.parametrize('header', ['Content-Length', 'content-length'])
    @pytest.mark.parametrize('chunked', [True, False])
    def test_skip_header_non_supported(self, header, chunked):
        with HTTPConnectionPool(self.host, self.port) as pool:
            with pytest.raises(ValueError) as e:
                pool.request('GET', '/headers', headers={header: SKIP_HEADER}, chunked=chunked)
            assert str(e.value) == "urllib3.util.SKIP_HEADER only supports 'Accept-Encoding', 'Host', 'User-Agent'"
            assert all(("'" + header.title() + "'" in str(e.value) for header in SKIPPABLE_HEADERS))

    @pytest.mark.parametrize('chunked', [True, False])
    @pytest.mark.parametrize('pool_request', [True, False])
    @pytest.mark.parametrize('header_type', [dict, HTTPHeaderDict])
    def test_headers_not_modified_by_request(self, chunked, pool_request, header_type):
        headers = header_type()
        headers['key'] = 'val'
        with HTTPConnectionPool(self.host, self.port) as pool:
            pool.headers = headers
            if pool_request:
                pool.request('GET', '/headers', chunked=chunked)
            else:
                conn = pool._get_conn()
                if chunked:
                    conn.request_chunked('GET', '/headers')
                else:
                    conn.request('GET', '/headers')
            assert pool.headers == {'key': 'val'}
            assert isinstance(pool.headers, header_type)
        with HTTPConnectionPool(self.host, self.port) as pool:
            if pool_request:
                pool.request('GET', '/headers', headers=headers, chunked=chunked)
            else:
                conn = pool._get_conn()
                if chunked:
                    conn.request_chunked('GET', '/headers', headers=headers)
                else:
                    conn.request('GET', '/headers', headers=headers)
            assert headers == {'key': 'val'}

    def test_bytes_header(self):
        with HTTPConnectionPool(self.host, self.port) as pool:
            headers = {'User-Agent': b'test header'}
            r = pool.request('GET', '/headers', headers=headers)
            request_headers = json.loads(r.data.decode('utf8'))
            assert 'User-Agent' in request_headers
            assert request_headers['User-Agent'] == 'test header'

    @pytest.mark.parametrize('user_agent', [u'Schönefeld/1.18.0', u'Schönefeld/1.18.0'.encode('iso-8859-1')])
    def test_user_agent_non_ascii_user_agent(self, user_agent):
        if six.PY2 and (not isinstance(user_agent, str)):
            pytest.skip('Python 2 raises UnicodeEncodeError when passed a unicode header')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.urlopen('GET', '/headers', headers={'User-Agent': user_agent})
            request_headers = json.loads(r.data.decode('utf8'))
            assert 'User-Agent' in request_headers
            assert request_headers['User-Agent'] == u'Schönefeld/1.18.0'

    @onlyPy2
    def test_user_agent_non_ascii_fails_on_python_2(self):
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with pytest.raises(UnicodeEncodeError) as e:
                pool.urlopen('GET', '/headers', headers={'User-Agent': u'Schönefeld/1.18.0'})
            assert str(e.value) == "'ascii' codec can't encode character u'\\xf6' in position 3: ordinal not in range(128)"