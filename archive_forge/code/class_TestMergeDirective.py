from breezy import merge_directive
from breezy.bzr import chk_map
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestMergeDirective(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def make_two_branches(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('f', b'f-id', 'file', b'initial content\n'))], revision_id=b'A')
        builder.build_snapshot([b'A'], [('modify', ('f', b'new content\n'))], revision_id=b'B')
        builder.finish_series()
        b1 = builder.get_branch()
        b2 = b1.controldir.sprout('target', revision_id=b'A').open_branch()
        return (b1, b2)

    def create_merge_directive(self, source_branch, submit_url):
        return merge_directive.MergeDirective2.from_objects(source_branch.repository, source_branch.last_revision(), time=1247775710, timezone=0, target_branch=submit_url)

    def test_create_merge_directive(self):
        source_branch, target_branch = self.make_two_branches()
        directive = self.create_merge_directive(source_branch, target_branch.base)
        self.assertIsInstance(directive, merge_directive.MergeDirective2)

    def test_create_and_install_directive(self):
        source_branch, target_branch = self.make_two_branches()
        directive = self.create_merge_directive(source_branch, target_branch.base)
        chk_map.clear_cache()
        directive.install_revisions(target_branch.repository)
        rt = target_branch.repository.revision_tree(b'B')
        with rt.lock_read():
            self.assertEqualDiff(b'new content\n', rt.get_file_text('f'))