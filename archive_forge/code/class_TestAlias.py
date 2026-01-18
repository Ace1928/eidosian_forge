from breezy import config, tests
from breezy.tests import features
class TestAlias(tests.TestCaseWithTransport):

    def test_list_alias_with_none(self):
        """Calling alias with no parameters lists existing aliases."""
        out, err = self.run_bzr('alias')
        self.assertEqual('', out)

    def test_list_unknown_alias(self):
        out, err = self.run_bzr('alias commit')
        self.assertEqual('brz alias: commit: not found\n', out)

    def test_add_alias_outputs_nothing(self):
        out, err = self.run_bzr('alias commit="commit --strict"')
        self.assertEqual('', out)

    def test_add_alias_visible(self):
        """Adding an alias makes it ..."""
        self.run_bzr('alias commit="commit --strict"')
        out, err = self.run_bzr('alias commit')
        self.assertEqual('brz alias commit="commit --strict"\n', out)

    def test_unicode_alias(self):
        """Unicode aliases should work (Bug #529930)"""
        self.requireFeature(features.UnicodeFilenameFeature)
        file_name = 'foo¶'
        tree = self.make_branch_and_tree('.')
        self.build_tree([file_name])
        tree.add(file_name)
        tree.commit('added')
        config.GlobalConfig.from_string('[ALIASES]\nust=st {}\n'.format(file_name), save=True)
        out, err = self.run_bzr('ust')
        self.assertEqual(err, '')
        self.assertEqual(out, '')

    def test_alias_listing_alphabetical(self):
        self.run_bzr('alias commit="commit --strict"')
        self.run_bzr('alias ll="log --short"')
        self.run_bzr('alias add="add -q"')
        out, err = self.run_bzr('alias')
        self.assertEqual('brz alias add="add -q"\nbrz alias commit="commit --strict"\nbrz alias ll="log --short"\n', out)

    def test_remove_unknown_alias(self):
        out, err = self.run_bzr('alias --remove fooix', retcode=3)
        self.assertEqual('brz: ERROR: The alias "fooix" does not exist.\n', err)

    def test_remove_known_alias(self):
        self.run_bzr('alias commit="commit --strict"')
        out, err = self.run_bzr('alias commit')
        self.assertEqual('brz alias commit="commit --strict"\n', out)
        out, err = self.run_bzr('alias --remove commit')
        self.assertEqual('', out)
        out, err = self.run_bzr('alias commit')
        self.assertEqual('brz alias: commit: not found\n', out)