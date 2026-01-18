import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
class TestGitDirFormat(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.format = dir.LocalGitControlDirFormat()

    def test_get_format_description(self):
        self.assertEqual('Local Git Repository', self.format.get_format_description())

    def test_eq(self):
        format2 = dir.LocalGitControlDirFormat()
        self.assertEqual(self.format, format2)
        self.assertEqual(self.format, self.format)
        bzr_format = controldir.format_registry.make_controldir('default')
        self.assertNotEqual(self.format, bzr_format)