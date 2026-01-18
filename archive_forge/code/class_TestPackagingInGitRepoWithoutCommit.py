import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
class TestPackagingInGitRepoWithoutCommit(base.BaseTestCase):

    def setUp(self):
        super(TestPackagingInGitRepoWithoutCommit, self).setUp()
        self.useFixture(TestRepo(self.package_dir))
        self.run_setup('sdist', allow_fail=False)

    def test_authors(self):
        with open(os.path.join(self.package_dir, 'AUTHORS'), 'r') as f:
            body = f.read()
        self.assertEqual('\n', body)

    def test_changelog(self):
        with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
            body = f.read()
        self.assertEqual('CHANGES\n=======\n\n', body)