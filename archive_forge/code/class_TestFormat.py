import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
class TestFormat(per_branch.TestCaseWithBranch):
    """Tests for the format itself."""

    def test_get_reference(self):
        """get_reference on all regular branches should return None."""
        if not self.branch_format.is_supported():
            return
        made_controldir = self.make_controldir('.')
        made_controldir.create_repository()
        if made_controldir._format.colocated_branches:
            name = 'foo'
        else:
            name = None
        try:
            made_branch = made_controldir.create_branch(name)
        except errors.UninitializableFormat:
            raise tests.TestNotApplicable('Uninitializable branch format')
        self.assertEqual(None, made_branch._format.get_reference(made_branch.controldir, name))

    def test_set_reference(self):
        """set_reference on all regular branches should be callable."""
        if not self.branch_format.is_supported():
            return
        this_branch = self.make_branch('this')
        other_branch = self.make_branch('other')
        try:
            this_branch._format.set_reference(this_branch.controldir, None, other_branch)
        except (NotImplementedError, errors.IncompatibleFormat):
            pass
        else:
            ref = this_branch._format.get_reference(this_branch.controldir)
            self.assertEqual(ref, other_branch.user_url)

    def test_format_initialize_find_open(self):
        if not self.branch_format.is_supported():
            return
        t = self.get_transport()
        readonly_t = transport.get_transport_from_url(self.get_readonly_url())
        made_branch = self.make_branch('.')
        self.assertIsInstance(made_branch, _mod_branch.Branch)
        opened_control = controldir.ControlDir.open(readonly_t.base)
        direct_opened_branch = opened_control.open_branch()
        self.assertEqual(direct_opened_branch.__class__, made_branch.__class__)
        self.assertEqual(opened_control, direct_opened_branch.controldir)
        self.assertIsInstance(direct_opened_branch._format, self.branch_format.__class__)
        opened_branch = _mod_branch.Branch.open(readonly_t.base)
        self.assertIsInstance(opened_branch, made_branch.__class__)
        self.assertEqual(made_branch._format.__class__, opened_branch._format.__class__)
        try:
            self.branch_format.get_format_string()
        except NotImplementedError:
            return
        self.assertEqual(self.branch_format, opened_control.find_branch_format())