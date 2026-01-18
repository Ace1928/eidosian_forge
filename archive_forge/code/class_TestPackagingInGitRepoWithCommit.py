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
class TestPackagingInGitRepoWithCommit(base.BaseTestCase):
    scenarios = [('preversioned', dict(preversioned=True)), ('postversioned', dict(preversioned=False))]

    def setUp(self):
        super(TestPackagingInGitRepoWithCommit, self).setUp()
        self.repo = self.useFixture(TestRepo(self.package_dir))
        self.repo.commit()

    def test_authors(self):
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'AUTHORS'), 'r') as f:
            body = f.read()
        self.assertNotEqual(body, '')

    def test_changelog(self):
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
            body = f.read()
        self.assertNotEqual(body, '')

    def test_changelog_handles_astrisk(self):
        self.repo.commit(message_content='Allow *.openstack.org to work')
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
            body = f.read()
        self.assertIn('\\*', body)

    def test_changelog_handles_dead_links_in_commit(self):
        self.repo.commit(message_content='See os_ for to_do about qemu_.')
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
            body = f.read()
        self.assertIn('os\\_', body)
        self.assertIn('to\\_do', body)
        self.assertIn('qemu\\_', body)

    def test_changelog_handles_backticks(self):
        self.repo.commit(message_content='Allow `openstack.org` to `work')
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
            body = f.read()
        self.assertIn('\\`', body)

    def test_manifest_exclude_honoured(self):
        self.run_setup('sdist', allow_fail=False)
        with open(os.path.join(self.package_dir, 'pbr_testpackage.egg-info/SOURCES.txt'), 'r') as f:
            body = f.read()
        self.assertThat(body, matchers.Not(matchers.Contains('pbr_testpackage/extra.py')))
        self.assertThat(body, matchers.Contains('pbr_testpackage/__init__.py'))

    def test_install_writes_changelog(self):
        stdout, _, _ = self.run_setup('install', '--root', self.temp_dir + 'installed', allow_fail=False)
        self.expectThat(stdout, matchers.Contains('Generating ChangeLog'))