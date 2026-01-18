import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
class TestBashCompletionInvoking(tests.TestCaseWithTransport, BashCompletionMixin):
    """Test bash completions that might execute brz.

    Only the syntax ``$(brz ...`` is supported so far. The brz command
    will be replaced by the brz instance running this selftest.
    """

    def setUp(self):
        super().setUp()
        if sys.platform == 'win32':
            raise tests.KnownFailure('see bug #709104, completion is broken on windows')

    def get_script(self):
        s = super().get_script()
        s = s.replace('$(brz ', '$(%s ' % ' '.join(self.get_brz_command()))
        s = s.replace('2>/dev/null', '')
        return s

    def test_revspec_tag_all(self):
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', b'null:')
        wt.branch.tags.set_tag('tag2', b'null:')
        wt.branch.tags.set_tag('3tag', b'null:')
        self.complete(['brz', 'log', '-r', 'tag', ':'])
        self.assertCompletionEquals('tag1', 'tag2', '3tag')

    def test_revspec_tag_prefix(self):
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', b'null:')
        wt.branch.tags.set_tag('tag2', b'null:')
        wt.branch.tags.set_tag('3tag', b'null:')
        self.complete(['brz', 'log', '-r', 'tag', ':', 't'])
        self.assertCompletionEquals('tag1', 'tag2')

    def test_revspec_tag_spaces(self):
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag with spaces', b'null:')
        self.complete(['brz', 'log', '-r', 'tag', ':', 't'])
        self.assertCompletionEquals('tag\\ with\\ spaces')
        self.complete(['brz', 'log', '-r', '"tag:t'])
        self.assertCompletionEquals('tag:tag with spaces')
        self.complete(['brz', 'log', '-r', "'tag:t"])
        self.assertCompletionEquals('tag:tag with spaces')

    def test_revspec_tag_endrange(self):
        self.requireFeature(features.sed_feature)
        wt = self.make_branch_and_tree('.', format='dirstate-tags')
        wt.branch.tags.set_tag('tag1', b'null:')
        wt.branch.tags.set_tag('tag2', b'null:')
        self.complete(['brz', 'log', '-r', '3..tag', ':', 't'])
        self.assertCompletionEquals('tag1', 'tag2')
        self.complete(['brz', 'log', '-r', '"3..tag:t'])
        self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')
        self.complete(['brz', 'log', '-r', "'3..tag:t"])
        self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')