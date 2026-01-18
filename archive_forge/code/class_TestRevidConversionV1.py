from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
class TestRevidConversionV1(tests.TestCase):

    def test_simple_git_to_bzr_revision_id(self):
        self.assertEqual(b'git-v1:c6a4d8f1fa4ac650748e647c4b1b368f589a7356', BzrGitMappingv1().revision_id_foreign_to_bzr(b'c6a4d8f1fa4ac650748e647c4b1b368f589a7356'))

    def test_simple_bzr_to_git_revision_id(self):
        self.assertEqual((b'c6a4d8f1fa4ac650748e647c4b1b368f589a7356', BzrGitMappingv1()), BzrGitMappingv1().revision_id_bzr_to_foreign(b'git-v1:c6a4d8f1fa4ac650748e647c4b1b368f589a7356'))

    def test_is_control_file(self):
        mapping = BzrGitMappingv1()
        if mapping.roundtripping:
            self.assertTrue(mapping.is_special_file('.bzrdummy'))
            self.assertTrue(mapping.is_special_file('.bzrfileids'))
        self.assertFalse(mapping.is_special_file('.bzrfoo'))

    def test_generate_file_id(self):
        mapping = BzrGitMappingv1()
        self.assertIsInstance(mapping.generate_file_id('la'), bytes)
        self.assertIsInstance(mapping.generate_file_id('é'), bytes)