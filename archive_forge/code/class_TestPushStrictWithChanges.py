import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class TestPushStrictWithChanges(tests.TestCaseWithTransport, TestPushStrictMixin):
    scenarios = strict_push_change_scenarios
    _changes_type = None

    def setUp(self):
        super().setUp()
        getattr(self, self._changes_type)()

    def _uncommitted_changes(self):
        self.make_local_branch_and_tree()
        self.build_tree_contents([('local/file', b'in progress')])

    def _pending_merges(self):
        self.make_local_branch_and_tree()
        other_bzrdir = self.tree.controldir.sprout('other')
        other_tree = other_bzrdir.open_workingtree()
        self.build_tree_contents([('other/other-file', b'other')])
        other_tree.add('other-file')
        other_tree.commit('other commit', rev_id=b'other')
        self.tree.merge_from_branch(other_tree.branch)
        self.tree.revert(filenames=['other-file'], backups=False)

    def _out_of_sync_trees(self):
        self.make_local_branch_and_tree()
        self.run_bzr(['checkout', '--lightweight', 'local', 'checkout'])
        self.build_tree_contents([('local/file', b'modified in local')])
        self.tree.commit('modify file', rev_id=b'modified-in-local')
        self._default_wd = 'checkout'
        self._default_errors = ["Working tree is out of date, please run 'brz update'\\."]

    def test_push_default(self):
        self.assertPushSucceeds([], with_warning=True)

    def test_push_with_revision(self):
        self.assertPushSucceeds(['-r', 'revid:added'], revid_to_push=b'added')

    def test_push_no_strict(self):
        self.assertPushSucceeds(['--no-strict'])

    def test_push_strict_with_changes(self):
        self.assertPushFails(['--strict'])

    def test_push_respect_config_var_strict(self):
        self.set_config_push_strict('true')
        self.assertPushFails([])

    def test_push_bogus_config_var_ignored(self):
        self.set_config_push_strict("I don't want you to be strict")
        self.assertPushSucceeds([], with_warning=True)

    def test_push_no_strict_command_line_override_config(self):
        self.set_config_push_strict('yES')
        self.assertPushFails([])
        self.assertPushSucceeds(['--no-strict'])

    def test_push_strict_command_line_override_config(self):
        self.set_config_push_strict('oFF')
        self.assertPushFails(['--strict'])
        self.assertPushSucceeds([])