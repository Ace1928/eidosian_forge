from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
class TestSendStrictMixin(TestSendMixin):

    def make_parent_and_local_branches(self):
        self.parent_tree = ControlDir.create_standalone_workingtree('parent')
        self.build_tree_contents([('parent/file', b'parent')])
        self.parent_tree.add('file')
        parent = self.parent_tree.commit('first commit')
        local_bzrdir = self.parent_tree.controldir.sprout('local')
        self.local_tree = local_bzrdir.open_workingtree()
        self.build_tree_contents([('local/file', b'local')])
        local = self.local_tree.commit('second commit')
        return (parent, local)
    _default_command = ['send', '-o-', '../parent']
    _default_wd = 'local'
    _default_sent_revs = None
    _default_errors = ['Working tree ".*/local/" has uncommitted changes \\(See brz status\\)\\.']
    _default_additional_error = 'Use --no-strict to force the send.\n'
    _default_additional_warning = 'Uncommitted changes will not be sent.'

    def set_config_send_strict(self, value):
        br = branch.Branch.open('local')
        br.get_config_stack().set('send_strict', value)

    def assertSendFails(self, args):
        out, err = self.run_send(args, rc=3, err_re=self._default_errors)
        self.assertContainsRe(err, self._default_additional_error)

    def assertSendSucceeds(self, args, revs=None, with_warning=False):
        if with_warning:
            err_re = self._default_errors
        else:
            err_re = []
        if revs is None:
            revs = self._default_sent_revs or [self.local]
        out, err = self.run_send(args, err_re=err_re)
        if len(revs) == 1:
            bundling_revs = 'Bundling %d revision.\n' % len(revs)
        else:
            bundling_revs = 'Bundling %d revisions.\n' % len(revs)
        if with_warning:
            self.assertContainsRe(err, self._default_additional_warning)
            self.assertEndsWith(err, bundling_revs)
        else:
            self.assertEqual(bundling_revs, err)
        md = merge_directive.MergeDirective.from_lines(BytesIO(out.encode('utf-8')))
        self.assertEqual(self.parent, md.base_revision_id)
        br = serializer.read_bundle(BytesIO(md.get_raw_bundle()))
        self.assertEqual(set(revs), {r.revision_id for r in br.revisions})