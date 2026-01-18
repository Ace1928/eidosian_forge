import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
@skipIf(missing_libs, skipmsg)
class TestSwiftRepo(TestCase):

    def setUp(self):
        super().setUp()
        self.conf = swift.load_conf(file=StringIO(config_file % def_config_file))

    def test_init(self):
        store = {'fakerepo/objects/pack': ''}
        with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector, store=store):
            swift.SwiftRepo('fakerepo', conf=self.conf)

    def test_init_no_data(self):
        with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector):
            self.assertRaises(Exception, swift.SwiftRepo, 'fakerepo', self.conf)

    def test_init_bad_data(self):
        store = {'fakerepo/.git/objects/pack': ''}
        with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector, store=store):
            self.assertRaises(Exception, swift.SwiftRepo, 'fakerepo', self.conf)

    def test_put_named_file(self):
        store = {'fakerepo/objects/pack': ''}
        with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector, store=store):
            repo = swift.SwiftRepo('fakerepo', conf=self.conf)
            desc = b'Fake repo'
            repo._put_named_file('description', desc)
        self.assertEqual(repo.scon.store['fakerepo/description'], desc)

    def test_init_bare(self):
        fsc = FakeSwiftConnector('fakeroot', conf=self.conf)
        with patch('dulwich.contrib.swift.SwiftConnector', new_callable=create_swift_connector, store=fsc.store):
            swift.SwiftRepo.init_bare(fsc, conf=self.conf)
        self.assertIn('fakeroot/objects/pack', fsc.store)
        self.assertIn('fakeroot/info/refs', fsc.store)
        self.assertIn('fakeroot/description', fsc.store)