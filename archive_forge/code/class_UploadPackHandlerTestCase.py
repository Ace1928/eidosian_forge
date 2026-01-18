import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class UploadPackHandlerTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)
        self._repo = Repo.init_bare(self.path)
        backend = DictBackend({b'/': self._repo})
        self._handler = UploadPackHandler(backend, [b'/', b'host=lolcathost'], TestProto())

    def test_progress(self):
        caps = self._handler.required_capabilities()
        self._handler.set_client_capabilities(caps)
        self._handler._start_pack_send_phase()
        self._handler.progress(b'first message')
        self._handler.progress(b'second message')
        self.assertEqual(b'first message', self._handler.proto.get_received_line(2))
        self.assertEqual(b'second message', self._handler.proto.get_received_line(2))
        self.assertRaises(IndexError, self._handler.proto.get_received_line, 2)

    def test_no_progress(self):
        caps = [*list(self._handler.required_capabilities()), b'no-progress']
        self._handler.set_client_capabilities(caps)
        self._handler.progress(b'first message')
        self._handler.progress(b'second message')
        self.assertRaises(IndexError, self._handler.proto.get_received_line, 2)

    def test_get_tagged(self):
        refs = {b'refs/tags/tag1': ONE, b'refs/tags/tag2': TWO, b'refs/heads/master': FOUR}
        self._repo.object_store.add_object(make_commit(id=FOUR))
        for name, sha in refs.items():
            self._repo.refs[name] = sha
        peeled = {b'refs/tags/tag1': b'1234' * 10, b'refs/tags/tag2': b'5678' * 10}
        self._repo.refs._peeled_refs = peeled
        self._repo.refs.add_packed_refs(refs)
        caps = [*list(self._handler.required_capabilities()), b'include-tag']
        self._handler.set_client_capabilities(caps)
        self.assertEqual({b'1234' * 10: ONE, b'5678' * 10: TWO}, self._handler.get_tagged(refs, repo=self._repo))
        caps = self._handler.required_capabilities()
        self._handler.set_client_capabilities(caps)
        self.assertEqual({}, self._handler.get_tagged(refs, repo=self._repo))

    def test_nothing_to_do_but_wants(self):
        refs = {b'refs/tags/tag1': ONE}
        tree = Tree()
        self._repo.object_store.add_object(tree)
        self._repo.object_store.add_object(make_commit(id=ONE, tree=tree))
        for name, sha in refs.items():
            self._repo.refs[name] = sha
        self._handler.proto.set_output([b'want ' + ONE + b' side-band-64k thin-pack ofs-delta', None, b'have ' + ONE, b'done', None])
        self._handler.handle()
        self.assertTrue(self._handler.proto.get_received_line(1).startswith(b'PACK'))

    def test_nothing_to_do_no_wants(self):
        refs = {b'refs/tags/tag1': ONE}
        tree = Tree()
        self._repo.object_store.add_object(tree)
        self._repo.object_store.add_object(make_commit(id=ONE, tree=tree))
        for ref, sha in refs.items():
            self._repo.refs[ref] = sha
        self._handler.proto.set_output([None])
        self._handler.handle()
        self.assertEqual([], self._handler.proto._received[1])