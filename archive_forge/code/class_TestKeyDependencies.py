from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
class TestKeyDependencies(TestCaseWithTransport):

    def get_format(self):
        return controldir.format_registry.make_controldir(self.format_name)

    def create_source_and_target(self):
        builder = self.make_branch_builder('source', format=self.get_format())
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id', b'ghost-id'], [], revision_id=b'B-id')
        builder.finish_series()
        repo = self.make_repository('target', format=self.get_format())
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        return (b.repository, repo)

    def test_key_dependencies_cleared_on_abort(self):
        source_repo, target_repo = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            stream = source_repo.revisions.get_record_stream([(b'B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([(b'B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.abort_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))

    def test_key_dependencies_cleared_on_suspend(self):
        source_repo, target_repo = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            stream = source_repo.revisions.get_record_stream([(b'B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([(b'B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.suspend_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))

    def test_key_dependencies_cleared_on_commit(self):
        source_repo, target_repo = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            for vf_name in ['texts', 'chk_bytes', 'inventories']:
                source_vf = getattr(source_repo, vf_name, None)
                if source_vf is None:
                    continue
                target_vf = getattr(target_repo, vf_name)
                stream = source_vf.get_record_stream(source_vf.keys(), 'unordered', True)
                target_vf.insert_record_stream(stream)
            stream = source_repo.revisions.get_record_stream([(b'B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([(b'B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.commit_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))