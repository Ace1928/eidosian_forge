import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class TestActivityMixin:
    """Test socket activity reporting.

    We use a special purpose server to control the bytes sent and received and
    be able to predict the activity on the client socket.
    """

    def setUp(self):
        self.server = self._activity_server(self._protocol_version)
        self.server.start_server()
        self.addCleanup(self.server.stop_server)
        _activities = {}

        def report_activity(t, bytes, direction):
            count = _activities.get(direction, 0)
            count += bytes
            _activities[direction] = count
        self.activities = _activities
        self.overrideAttr(self._transport, '_report_activity', report_activity)

    def get_transport(self):
        t = self._transport(self.server.get_url())
        return t

    def assertActivitiesMatch(self):
        self.assertEqual(self.server.bytes_read, self.activities.get('write', 0), 'written bytes')
        self.assertEqual(self.server.bytes_written, self.activities.get('read', 0), 'read bytes')

    def test_get(self):
        self.server.canned_response = b'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: 35\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\nBazaar-NG meta directory, format 1\n'
        t = self.get_transport()
        self.assertEqual(b'Bazaar-NG meta directory, format 1\n', t.get('foo/bar').read())
        self.assertActivitiesMatch()

    def test_has(self):
        self.server.canned_response = b'HTTP/1.1 200 OK\r\nServer: SimpleHTTP/0.6 Python/2.5.2\r\nDate: Thu, 29 Jan 2009 20:21:47 GMT\r\nContent-type: application/octet-stream\r\nContent-Length: 20\r\nLast-Modified: Thu, 29 Jan 2009 20:21:47 GMT\r\n\r\n'
        t = self.get_transport()
        self.assertTrue(t.has('foo/bar'))
        self.assertActivitiesMatch()

    def test_readv(self):
        self.server.canned_response = b'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:49:48 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 1534\r\nConnection: close\r\nContent-Type: multipart/byteranges; boundary=418470f848b63279b\r\n\r\n\r\n--418470f848b63279b\r\nContent-type: text/plain; charset=UTF-8\r\nContent-range: bytes 0-254/93890\r\n\r\nmbp@sourcefrog.net-20050309040815-13242001617e4a06\nmbp@sourcefrog.net-20050309040929-eee0eb3e6d1e7627\nmbp@sourcefrog.net-20050309040957-6cad07f466bb0bb8\nmbp@sourcefrog.net-20050309041501-c840e09071de3b67\nmbp@sourcefrog.net-20050309044615-c24a3250be83220a\n\r\n--418470f848b63279b\r\nContent-type: text/plain; charset=UTF-8\r\nContent-range: bytes 1000-2049/93890\r\n\r\n40-fd4ec249b6b139ab\nmbp@sourcefrog.net-20050311063625-07858525021f270b\nmbp@sourcefrog.net-20050311231934-aa3776aff5200bb9\nmbp@sourcefrog.net-20050311231953-73aeb3a131c3699a\nmbp@sourcefrog.net-20050311232353-f5e33da490872c6a\nmbp@sourcefrog.net-20050312071639-0a8f59a34a024ff0\nmbp@sourcefrog.net-20050312073432-b2c16a55e0d6e9fb\nmbp@sourcefrog.net-20050312073831-a47c3335ece1920f\nmbp@sourcefrog.net-20050312085412-13373aa129ccbad3\nmbp@sourcefrog.net-20050313052251-2bf004cb96b39933\nmbp@sourcefrog.net-20050313052856-3edd84094687cb11\nmbp@sourcefrog.net-20050313053233-e30a4f28aef48f9d\nmbp@sourcefrog.net-20050313053853-7c64085594ff3072\nmbp@sourcefrog.net-20050313054757-a86c3f5871069e22\nmbp@sourcefrog.net-20050313061422-418f1f73b94879b9\nmbp@sourcefrog.net-20050313120651-497bd231b19df600\nmbp@sourcefrog.net-20050314024931-eae0170ef25a5d1a\nmbp@sourcefrog.net-20050314025438-d52099f915fe65fc\nmbp@sourcefrog.net-20050314025539-637a636692c055cf\nmbp@sourcefrog.net-20050314025737-55eb441f430ab4ba\nmbp@sourcefrog.net-20050314025901-d74aa93bb7ee8f62\nmbp@source\r\n--418470f848b63279b--\r\n'
        t = self.get_transport()
        l = list(t.readv('/foo/bar', ((0, 255), (1000, 1050))))
        t._get_connection().cleanup_pipe()
        self.assertEqual(2, len(l))
        self.assertActivitiesMatch()

    def test_post(self):
        self.server.canned_response = b'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: 35\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\nlalala whatever as long as itsssss\n'
        t = self.get_transport()
        code, f = t._post(b'abc def end-of-body\n')
        self.assertEqual(b'lalala whatever as long as itsssss\n', f.read())
        self.assertActivitiesMatch()