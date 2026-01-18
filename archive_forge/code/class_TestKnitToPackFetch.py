from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
class TestKnitToPackFetch(TestCaseWithTransport):

    def find_get_record_stream(self, calls, expected_count=1):
        """In a list of calls, find the last 'get_record_stream'.

        :param expected_count: The number of calls we should exepect to find.
            If a different number is found, an assertion is raised.
        """
        get_record_call = None
        call_count = 0
        for call in calls:
            if call[0] == 'get_record_stream':
                call_count += 1
                get_record_call = call
        self.assertEqual(expected_count, call_count)
        return get_record_call

    def test_fetch_with_deltas_no_delta_closure(self):
        tree = self.make_branch_and_tree('source', format='dirstate')
        target = self.make_repository('target', format='pack-0.92')
        self.build_tree(['source/file'])
        tree.set_root_id(b'root-id')
        tree.add('file', ids=b'file-id')
        tree.commit('one', rev_id=b'rev-one')
        source = tree.branch.repository
        source.texts = versionedfile.RecordingVersionedFilesDecorator(source.texts)
        source.signatures = versionedfile.RecordingVersionedFilesDecorator(source.signatures)
        source.revisions = versionedfile.RecordingVersionedFilesDecorator(source.revisions)
        source.inventories = versionedfile.RecordingVersionedFilesDecorator(source.inventories)
        self.assertTrue(target._format._fetch_uses_deltas)
        target.fetch(source, revision_id=b'rev-one')
        self.assertEqual(('get_record_stream', [(b'file-id', b'rev-one')], target._format._fetch_order, False), self.find_get_record_stream(source.texts.calls))
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(source.inventories.calls, 2))
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(source.revisions.calls))
        signature_calls = source.signatures.calls[-1:]
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(signature_calls))

    def test_fetch_no_deltas_with_delta_closure(self):
        tree = self.make_branch_and_tree('source', format='dirstate')
        target = self.make_repository('target', format='pack-0.92')
        self.build_tree(['source/file'])
        tree.set_root_id(b'root-id')
        tree.add('file', ids=b'file-id')
        tree.commit('one', rev_id=b'rev-one')
        source = tree.branch.repository
        source.texts = versionedfile.RecordingVersionedFilesDecorator(source.texts)
        source.signatures = versionedfile.RecordingVersionedFilesDecorator(source.signatures)
        source.revisions = versionedfile.RecordingVersionedFilesDecorator(source.revisions)
        source.inventories = versionedfile.RecordingVersionedFilesDecorator(source.inventories)
        self.overrideAttr(target._format, '_fetch_uses_deltas', False)
        target.fetch(source, revision_id=b'rev-one')
        self.assertEqual(('get_record_stream', [(b'file-id', b'rev-one')], target._format._fetch_order, True), self.find_get_record_stream(source.texts.calls))
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, True), self.find_get_record_stream(source.inventories.calls, 2))
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, True), self.find_get_record_stream(source.revisions.calls))
        signature_calls = source.signatures.calls[-1:]
        self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, True), self.find_get_record_stream(signature_calls))

    def test_fetch_revisions_with_deltas_into_pack(self):
        tree = self.make_branch_and_tree('source', format='dirstate')
        target = self.make_repository('target', format='pack-0.92')
        self.build_tree(['source/file'])
        tree.set_root_id(b'root-id')
        tree.add('file', ids=b'file-id')
        tree.commit('one', rev_id=b'rev-one')
        tree.branch.repository.revisions._max_delta_chain = 200
        tree.commit('two', rev_id=b'rev-two')
        source = tree.branch.repository
        source.lock_read()
        self.addCleanup(source.unlock)
        record = next(source.revisions.get_record_stream([(b'rev-two',)], 'unordered', False))
        self.assertEqual('knit-delta-gz', record.storage_kind)
        target.fetch(tree.branch.repository, revision_id=b'rev-two')
        target.lock_read()
        self.addCleanup(target.unlock)
        record = next(target.revisions.get_record_stream([(b'rev-two',)], 'unordered', False))
        self.assertEqual('knit-ft-gz', record.storage_kind)

    def test_fetch_with_fallback_and_merge(self):
        builder = self.make_branch_builder('source', format='pack-0.92')
        builder.start_series()
        to_add = [('add', ('', b'TREE_ROOT', 'directory', None))]
        for i in range(10):
            fname = 'file%03d' % (i,)
            fileid = ('%s-%s' % (fname, osutils.rand_chars(64))).encode('ascii')
            to_add.append(('add', (fname, fileid, 'file', b'content\n')))
        builder.build_snapshot(None, to_add, revision_id=b'A')
        builder.build_snapshot([b'A'], [], revision_id=b'B')
        builder.build_snapshot([b'A'], [], revision_id=b'C')
        builder.build_snapshot([b'C'], [], revision_id=b'D')
        builder.build_snapshot([b'D'], [], revision_id=b'E')
        builder.build_snapshot([b'E', b'B'], [], revision_id=b'F')
        builder.finish_series()
        source_branch = builder.get_branch()
        source_branch.controldir.sprout('base', revision_id=b'B')
        target_branch = self.make_branch('target', format='1.6')
        target_branch.set_stacked_on_url('../base')
        source = source_branch.repository
        source.lock_read()
        self.addCleanup(source.unlock)
        source.inventories = versionedfile.OrderingVersionedFilesDecorator(source.inventories, key_priority={(b'E',): 1, (b'D',): 2, (b'C',): 4, (b'F',): 3})
        records = [(record.key, record.storage_kind) for record in source.inventories.get_record_stream([(b'D',), (b'C',), (b'E',), (b'F',)], 'unordered', False)]
        self.assertEqual([((b'E',), 'knit-delta-gz'), ((b'D',), 'knit-delta-gz'), ((b'F',), 'knit-delta-gz'), ((b'C',), 'knit-delta-gz')], records)
        target_branch.lock_write()
        self.addCleanup(target_branch.unlock)
        target = target_branch.repository
        target.fetch(source, revision_id=b'F')
        stream = target.inventories.get_record_stream([(b'C',), (b'D',), (b'E',), (b'F',)], 'unordered', False)
        kinds = {record.key: record.storage_kind for record in stream}
        self.assertEqual({(b'C',): 'knit-ft-gz', (b'D',): 'knit-delta-gz', (b'E',): 'knit-delta-gz', (b'F',): 'knit-delta-gz'}, kinds)