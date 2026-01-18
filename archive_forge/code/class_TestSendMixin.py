from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
class TestSendMixin:
    _default_command = ['send', '-o-']
    _default_wd = 'branch'

    def run_send(self, args, cmd=None, rc=0, wd=None, err_re=None):
        if cmd is None:
            cmd = self._default_command
        if wd is None:
            wd = self._default_wd
        if err_re is None:
            err_re = []
        return self.run_bzr(cmd + args, retcode=rc, working_dir=wd, error_regexes=err_re)

    def get_MD(self, args, cmd=None, wd='branch'):
        md = self.run_send(args, cmd=cmd, wd=wd)[0]
        out = BytesIO(md.encode('utf-8'))
        return merge_directive.MergeDirective.from_lines(out)

    def assertBundleContains(self, revs, args, cmd=None, wd='branch'):
        md = self.get_MD(args, cmd=cmd, wd=wd)
        br = serializer.read_bundle(BytesIO(md.get_raw_bundle()))
        self.assertEqual(set(revs), {r.revision_id for r in br.revisions})