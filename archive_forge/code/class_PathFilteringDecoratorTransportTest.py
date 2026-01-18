import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class PathFilteringDecoratorTransportTest(tests.TestCase):
    """Pathfilter decoration specific tests."""

    def test_abspath(self):
        server = pathfilter.PathFilteringServer(transport.get_transport_from_url('memory:///foo/bar/'), lambda x: x)
        server.start_server()
        t = transport.get_transport_from_url(server.get_url())
        self.assertEqual(server.get_url(), t.abspath('/'))
        subdir_t = t.clone('subdir')
        self.assertEqual(server.get_url(), subdir_t.abspath('/'))
        server.stop_server()

    def make_pf_transport(self, filter_func=None):
        """Make a PathFilteringTransport backed by a MemoryTransport.

        :param filter_func: by default this will be a no-op function.  Use this
            parameter to override it."""
        if filter_func is None:

            def filter_func(x):
                return x
        server = pathfilter.PathFilteringServer(transport.get_transport_from_url('memory:///foo/bar/'), filter_func)
        server.start_server()
        self.addCleanup(server.stop_server)
        return transport.get_transport_from_url(server.get_url())

    def test__filter(self):
        t = self.make_pf_transport()
        self.assertEqual('foo', t._filter('foo'))
        self.assertEqual('foo/bar', t._filter('foo/bar'))
        self.assertEqual('', t._filter('..'))
        self.assertEqual('', t._filter('/'))
        t = t.clone('subdir1/subdir2')
        self.assertEqual('subdir1/subdir2/foo', t._filter('foo'))
        self.assertEqual('subdir1/subdir2/foo/bar', t._filter('foo/bar'))
        self.assertEqual('subdir1', t._filter('..'))
        self.assertEqual('', t._filter('/'))

    def test_filter_invocation(self):
        filter_log = []

        def filter(path):
            filter_log.append(path)
            return path
        t = self.make_pf_transport(filter)
        t.has('abc')
        self.assertEqual(['abc'], filter_log)
        del filter_log[:]
        t.clone('abc').has('xyz')
        self.assertEqual(['abc/xyz'], filter_log)
        del filter_log[:]
        t.has('/abc')
        self.assertEqual(['abc'], filter_log)

    def test_clone(self):
        t = self.make_pf_transport()
        relpath_cloned = t.clone('foo')
        abspath_cloned = t.clone('/foo')
        self.assertEqual(t.server, relpath_cloned.server)
        self.assertEqual(t.server, abspath_cloned.server)

    def test_url_preserves_pathfiltering(self):
        """Calling get_transport on a pathfiltered transport's base should
        produce a transport with exactly the same behaviour as the original
        pathfiltered transport.

        This is so that it is not possible to escape (accidentally or
        otherwise) the filtering by doing::
            url = filtered_transport.base
            parent_url = urlutils.join(url, '..')
            new_t = transport.get_transport_from_url(parent_url)
        """
        t = self.make_pf_transport()
        new_t = transport.get_transport_from_url(t.base)
        self.assertEqual(t.server, new_t.server)
        self.assertEqual(t.base, new_t.base)