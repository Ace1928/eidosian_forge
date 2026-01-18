from __future__ import print_function
import os
import fixtures
from pbr import git
from pbr import options
from pbr.tests import base
class SkipFileWrites(base.BaseTestCase):
    scenarios = [('changelog_option_true', dict(option_key='skip_changelog', option_value='True', env_key='SKIP_WRITE_GIT_CHANGELOG', env_value=None, pkg_func=git.write_git_changelog, filename='ChangeLog')), ('changelog_option_false', dict(option_key='skip_changelog', option_value='False', env_key='SKIP_WRITE_GIT_CHANGELOG', env_value=None, pkg_func=git.write_git_changelog, filename='ChangeLog')), ('changelog_env_true', dict(option_key='skip_changelog', option_value='False', env_key='SKIP_WRITE_GIT_CHANGELOG', env_value='True', pkg_func=git.write_git_changelog, filename='ChangeLog')), ('changelog_both_true', dict(option_key='skip_changelog', option_value='True', env_key='SKIP_WRITE_GIT_CHANGELOG', env_value='True', pkg_func=git.write_git_changelog, filename='ChangeLog')), ('authors_option_true', dict(option_key='skip_authors', option_value='True', env_key='SKIP_GENERATE_AUTHORS', env_value=None, pkg_func=git.generate_authors, filename='AUTHORS')), ('authors_option_false', dict(option_key='skip_authors', option_value='False', env_key='SKIP_GENERATE_AUTHORS', env_value=None, pkg_func=git.generate_authors, filename='AUTHORS')), ('authors_env_true', dict(option_key='skip_authors', option_value='False', env_key='SKIP_GENERATE_AUTHORS', env_value='True', pkg_func=git.generate_authors, filename='AUTHORS')), ('authors_both_true', dict(option_key='skip_authors', option_value='True', env_key='SKIP_GENERATE_AUTHORS', env_value='True', pkg_func=git.generate_authors, filename='AUTHORS'))]

    def setUp(self):
        super(SkipFileWrites, self).setUp()
        self.temp_path = self.useFixture(fixtures.TempDir()).path
        self.root_dir = os.path.abspath(os.path.curdir)
        self.git_dir = os.path.join(self.root_dir, '.git')
        if not os.path.exists(self.git_dir):
            self.skipTest('%s is missing; skipping git-related checks' % self.git_dir)
            return
        self.filename = os.path.join(self.temp_path, self.filename)
        self.option_dict = dict()
        if self.option_key is not None:
            self.option_dict[self.option_key] = ('setup.cfg', self.option_value)
        self.useFixture(fixtures.EnvironmentVariable(self.env_key, self.env_value))

    def test_skip(self):
        self.pkg_func(git_dir=self.git_dir, dest_dir=self.temp_path, option_dict=self.option_dict)
        self.assertEqual(not os.path.exists(self.filename), self.option_value.lower() in options.TRUE_VALUES or self.env_value is not None)