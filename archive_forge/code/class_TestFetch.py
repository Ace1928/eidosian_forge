from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
class TestFetch(TestFetchBase):

    def test_sprout_from_stacked_with_short_history(self):
        content, source_b = self.make_source_branch()
        stack_b = self.make_branch('stack-on')
        stack_b.pull(source_b, stop_revision=b'B-id')
        target_b = self.make_branch('target')
        target_b.set_stacked_on_url('../stack-on')
        target_b.pull(source_b, stop_revision=b'C-id')
        final_b = self.make_branch('final')
        final_b.pull(target_b)
        final_b.lock_read()
        self.addCleanup(final_b.unlock)
        self.assertEqual(b'C-id', final_b.last_revision())
        text_keys = [(b'a-id', b'A-id'), (b'a-id', b'B-id'), (b'a-id', b'C-id')]
        stream = final_b.repository.texts.get_record_stream(text_keys, 'unordered', True)
        records = sorted([(r.key, r.get_bytes_as('fulltext')) for r in stream])
        self.assertEqual([((b'a-id', b'A-id'), b''.join(content[:-2])), ((b'a-id', b'B-id'), b''.join(content[:-1])), ((b'a-id', b'C-id'), b''.join(content))], records)

    def test_sprout_from_smart_stacked_with_short_history(self):
        content, source_b = self.make_source_branch()
        transport = self.make_smart_server('server')
        transport.ensure_base()
        url = transport.abspath('')
        stack_b = source_b.controldir.sprout(url + '/stack-on', revision_id=b'B-id')
        target_transport = transport.clone('target')
        target_transport.ensure_base()
        target_bzrdir = self.bzrdir_format.initialize_on_transport(target_transport)
        target_bzrdir.create_repository()
        target_b = target_bzrdir.create_branch()
        target_b.set_stacked_on_url('../stack-on')
        target_b.pull(source_b, stop_revision=b'C-id')
        final_b = target_b.controldir.sprout('final').open_branch()
        self.assertEqual(b'C-id', final_b.last_revision())
        final2_b = target_b.controldir.sprout('final2', revision_id=b'C-id').open_branch()
        self.assertEqual(b'C-id', final_b.last_revision())

    def make_source_with_ghost_and_stacked_target(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id', b'ghost-id'], [], revision_id=b'B-id')
        builder.finish_series()
        source_b = builder.get_branch()
        source_b.lock_read()
        self.addCleanup(source_b.unlock)
        base = self.make_branch('base')
        base.pull(source_b, stop_revision=b'A-id')
        stacked = self.make_branch('stacked')
        stacked.set_stacked_on_url('../base')
        return (source_b, base, stacked)

    def test_fetch_with_ghost_stacked(self):
        source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
        stacked.pull(source_b, stop_revision=b'B-id')

    def test_fetch_into_smart_stacked_with_ghost(self):
        source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
        trans = self.make_smart_server('stacked')
        stacked = branch.Branch.open(trans.base)
        stacked.lock_write()
        self.addCleanup(stacked.unlock)
        stacked.pull(source_b, stop_revision=b'B-id')

    def test_fetch_to_stacked_from_smart_with_ghost(self):
        source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
        trans = self.make_smart_server('source')
        source_b = branch.Branch.open(trans.base)
        source_b.lock_read()
        self.addCleanup(source_b.unlock)
        stacked.pull(source_b, stop_revision=b'B-id')