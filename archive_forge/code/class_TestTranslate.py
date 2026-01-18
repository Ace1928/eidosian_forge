import io
from .. import errors, i18n, tests, workingtree
class TestTranslate(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.overrideAttr(i18n, '_translations', ZzzTranslations())

    def test_error_message_translation(self):
        """do errors get translated?"""
        err = None
        tree = self.make_branch_and_tree('.')
        try:
            workingtree.WorkingTree.open('./foo')
        except errors.NotBranchError as e:
            err = str(e)
        self.assertContainsRe(err, 'zzå{{Not a branch: .*}}')

    def test_topic_help_translation(self):
        """does topic help get translated?"""
        from .. import help
        out = io.StringIO()
        help.help('authentication', out)
        self.assertContainsRe(out.getvalue(), 'zzå{{Authentication Settings')