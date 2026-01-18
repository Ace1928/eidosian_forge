import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class TestGetTransportAndPathFromUrl(TestCase):

    def test_tcp(self):
        c, path = get_transport_and_path_from_url('git://foo.com/bar/baz')
        self.assertIsInstance(c, TCPGitClient)
        self.assertEqual('foo.com', c._host)
        self.assertEqual(TCP_GIT_PORT, c._port)
        self.assertEqual('/bar/baz', path)

    def test_tcp_port(self):
        c, path = get_transport_and_path_from_url('git://foo.com:1234/bar/baz')
        self.assertIsInstance(c, TCPGitClient)
        self.assertEqual('foo.com', c._host)
        self.assertEqual(1234, c._port)
        self.assertEqual('/bar/baz', path)

    def test_ssh_explicit(self):
        c, path = get_transport_and_path_from_url('git+ssh://foo.com/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/bar/baz', path)

    def test_ssh_port_explicit(self):
        c, path = get_transport_and_path_from_url('git+ssh://foo.com:1234/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(1234, c.port)
        self.assertEqual('/bar/baz', path)

    def test_ssh_homepath(self):
        c, path = get_transport_and_path_from_url('git+ssh://foo.com/~/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(None, c.port)
        self.assertEqual(None, c.username)
        self.assertEqual('/~/bar/baz', path)

    def test_ssh_port_homepath(self):
        c, path = get_transport_and_path_from_url('git+ssh://foo.com:1234/~/bar/baz')
        self.assertIsInstance(c, SSHGitClient)
        self.assertEqual('foo.com', c.host)
        self.assertEqual(1234, c.port)
        self.assertEqual('/~/bar/baz', path)

    def test_ssh_host_relpath(self):
        self.assertRaises(ValueError, get_transport_and_path_from_url, 'foo.com:bar/baz')

    def test_ssh_user_host_relpath(self):
        self.assertRaises(ValueError, get_transport_and_path_from_url, 'user@foo.com:bar/baz')

    def test_local_path(self):
        self.assertRaises(ValueError, get_transport_and_path_from_url, 'foo.bar/baz')

    def test_error(self):
        self.assertRaises(ValueError, get_transport_and_path_from_url, 'prospero://bar/baz')

    def test_http(self):
        url = 'https://github.com/jelmer/dulwich'
        c, path = get_transport_and_path_from_url(url)
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('https://github.com', c.get_url(b'/'))
        self.assertEqual('/jelmer/dulwich', path)

    def test_http_port(self):
        url = 'https://github.com:9090/jelmer/dulwich'
        c, path = get_transport_and_path_from_url(url)
        self.assertEqual('https://github.com:9090', c.get_url(b'/'))
        self.assertIsInstance(c, HttpGitClient)
        self.assertEqual('/jelmer/dulwich', path)

    @patch('os.name', 'posix')
    @patch('sys.platform', 'linux')
    def test_file(self):
        c, path = get_transport_and_path_from_url('file:///home/jelmer/foo')
        self.assertIsInstance(c, LocalGitClient)
        self.assertEqual('/home/jelmer/foo', path)

    @patch('os.name', 'nt')
    @patch('sys.platform', 'win32')
    def test_file_win(self):
        from nturl2path import url2pathname
        with patch('dulwich.client.url2pathname', url2pathname):
            expected = 'C:\\foo.bar\\baz'
            for file_url in ['file:C:/foo.bar/baz', 'file:/C:/foo.bar/baz', 'file://C:/foo.bar/baz', 'file://C://foo.bar//baz', 'file:///C:/foo.bar/baz']:
                c, path = get_transport_and_path(file_url)
                self.assertIsInstance(c, LocalGitClient)
                self.assertEqual(path, expected)
            for remote_url in ['file://host.example.com/C:/foo.bar/bazfile://host.example.com/C:/foo.bar/bazfile:////host.example/foo.bar/baz']:
                with self.assertRaises(NotImplementedError):
                    c, path = get_transport_and_path(remote_url)