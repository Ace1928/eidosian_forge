import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
@skipIf(missing_libs, skipmsg)
class TestSwiftInfoRefsContainer(TestCase):

    def setUp(self):
        super().setUp()
        content = b'22effb216e3a82f97da599b8885a6cadb488b4c5\trefs/heads/master\ncca703b0e1399008b53a1a236d6b4584737649e4\trefs/heads/dev'
        self.store = {'fakerepo/info/refs': content}
        self.conf = swift.load_conf(file=StringIO(config_file % def_config_file))
        self.fsc = FakeSwiftConnector('fakerepo', conf=self.conf)
        self.object_store = {}

    def test_init(self):
        """info/refs does not exists."""
        irc = swift.SwiftInfoRefsContainer(self.fsc, self.object_store)
        self.assertEqual(len(irc._refs), 0)
        self.fsc.store = self.store
        irc = swift.SwiftInfoRefsContainer(self.fsc, self.object_store)
        self.assertIn(b'refs/heads/dev', irc.allkeys())
        self.assertIn(b'refs/heads/master', irc.allkeys())

    def test_set_if_equals(self):
        self.fsc.store = self.store
        irc = swift.SwiftInfoRefsContainer(self.fsc, self.object_store)
        irc.set_if_equals(b'refs/heads/dev', b'cca703b0e1399008b53a1a236d6b4584737649e4', b'1' * 40)
        self.assertEqual(irc[b'refs/heads/dev'], b'1' * 40)

    def test_remove_if_equals(self):
        self.fsc.store = self.store
        irc = swift.SwiftInfoRefsContainer(self.fsc, self.object_store)
        irc.remove_if_equals(b'refs/heads/dev', b'cca703b0e1399008b53a1a236d6b4584737649e4')
        self.assertNotIn(b'refs/heads/dev', irc.allkeys())