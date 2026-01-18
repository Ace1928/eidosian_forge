import gzip
import os
import re
import tempfile
from .... import tests
from ....tests import features
from ....tests.blackbox import ExternalBase
from ..cmds import _get_source_stream
from . import FastimportFeature
from :1
from :2
from :1
from :2
class TestFastExport(ExternalBase):
    _test_needs_features = [FastimportFeature]

    def test_empty(self):
        self.make_branch_and_tree('br')
        self.assertEqual('', self.run_bzr('fast-export br')[0])

    def test_pointless(self):
        tree = self.make_branch_and_tree('br')
        tree.commit('pointless')
        data = self.run_bzr('fast-export br')[0]
        self.assertTrue(data.startswith('reset refs/heads/master\ncommit refs/heads/master\nmark :1\ncommitter'), data)

    def test_file(self):
        tree = self.make_branch_and_tree('br')
        tree.commit('pointless')
        data = self.run_bzr('fast-export br br.fi')[0]
        self.assertEqual('', data)
        self.assertPathExists('br.fi')

    def test_symlink(self):
        tree = self.make_branch_and_tree('br')
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        os.symlink('symlink-target', 'br/symlink')
        tree.add('symlink')
        tree.commit('add a symlink')
        data = self.run_bzr('fast-export br br.fi')[0]
        self.assertEqual('', data)
        self.assertPathExists('br.fi')

    def test_tag_rewriting(self):
        tree = self.make_branch_and_tree('br')
        tree.commit('pointless')
        self.assertTrue(tree.branch.supports_tags())
        rev_id = tree.branch.dotted_revno_to_revision_id((1,))
        tree.branch.tags.set_tag('goodTag', rev_id)
        tree.branch.tags.set_tag('bad Tag', rev_id)
        data = self.run_bzr('fast-export --plain --no-rewrite-tag-names br')[0]
        self.assertNotEqual(-1, data.find('reset refs/tags/goodTag'))
        self.assertEqual(data.find('reset refs/tags/'), data.rfind('reset refs/tags/'))
        data = self.run_bzr('fast-export --plain --rewrite-tag-names br')[0]
        self.assertNotEqual(-1, data.find('reset refs/tags/goodTag'))
        self.assertNotEqual(-1, data.find('reset refs/tags/bad_Tag'))

    def test_no_tags(self):
        tree = self.make_branch_and_tree('br')
        tree.commit('pointless')
        self.assertTrue(tree.branch.supports_tags())
        rev_id = tree.branch.dotted_revno_to_revision_id((1,))
        tree.branch.tags.set_tag('someTag', rev_id)
        data = self.run_bzr('fast-export --plain --no-tags br')[0]
        self.assertEqual(-1, data.find('reset refs/tags/someTag'))

    def test_baseline_option(self):
        tree = self.make_branch_and_tree('bl')
        with open('bl/a', 'w') as f:
            f.write('test 1')
        tree.add('a')
        tree.commit(message='add a')
        with open('bl/b', 'w') as f:
            f.write('test 2')
        with open('bl/a', 'a') as f:
            f.write('\ntest 3')
        tree.add('b')
        tree.commit(message='add b, modify a')
        with open('bl/c', 'w') as f:
            f.write('test 4')
        tree.add('c')
        tree.remove('b')
        tree.commit(message='add c, remove b')
        with open('bl/a', 'a') as f:
            f.write('\ntest 5')
        tree.commit(message='modify a again')
        with open('bl/d', 'w') as f:
            f.write('test 6')
        tree.add('d')
        tree.commit(message='add d')
        data = self.run_bzr('fast-export --baseline -r 3.. bl')[0]
        data = re.sub('committer.*', 'committer', data)
        self.assertIn(data, (fast_export_baseline_data1, fast_export_baseline_data2))
        data1 = self.run_bzr('fast-export --baseline bl')[0]
        data2 = self.run_bzr('fast-export bl')[0]
        self.assertEqual(data1, data2)