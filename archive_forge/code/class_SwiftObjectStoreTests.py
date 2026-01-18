import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
@skipIf(missing_libs, skipmsg)
class SwiftObjectStoreTests(ObjectStoreTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        conf = swift.load_conf(file=StringIO(config_file % def_config_file))
        fsc = FakeSwiftConnector('fakerepo', conf=conf)
        self.store = swift.SwiftObjectStore(fsc)