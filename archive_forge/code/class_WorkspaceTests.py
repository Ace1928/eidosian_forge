import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
class WorkspaceTests(TestCaseWithTransport):
    scenarios = multiply_scenarios(vary_by_inotify(), vary_by_format())

    def setUp(self):
        super().setUp()
        if self._use_inotify:
            self.requireFeature(features.pyinotify)

    def test_root_add(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        with Workspace(tree, use_inotify=self._use_inotify) as ws:
            self.build_tree_contents([('afile', 'somecontents')])
            changes = [c for c in ws.iter_changes() if c.path[1] != '']
            self.assertEqual(1, len(changes), changes)
            self.assertEqual((None, 'afile'), changes[0].path)
            ws.commit(message='Commit message')
            self.assertEqual(list(ws.iter_changes()), [])
            self.build_tree_contents([('afile', 'newcontents')])
            [change] = list(ws.iter_changes())
            self.assertEqual(('afile', 'afile'), change.path)

    def test_root_remove(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        self.build_tree_contents([('afile', 'somecontents')])
        tree.add(['afile'])
        tree.commit('Afile')
        with Workspace(tree, use_inotify=self._use_inotify) as ws:
            os.remove('afile')
            changes = list(ws.iter_changes())
            self.assertEqual(1, len(changes), changes)
            self.assertEqual(('afile', None), changes[0].path)
            ws.commit(message='Commit message')
            self.assertEqual(list(ws.iter_changes()), [])

    def test_subpath_add(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        self.build_tree(['subpath/'])
        tree.add('subpath')
        tree.commit('add subpath')
        with Workspace(tree, subpath='subpath', use_inotify=self._use_inotify) as ws:
            self.build_tree_contents([('outside', 'somecontents')])
            self.build_tree_contents([('subpath/afile', 'somecontents')])
            changes = [c for c in ws.iter_changes() if c.path[1] != 'subpath']
            self.assertEqual(1, len(changes), changes)
            self.assertEqual((None, 'subpath/afile'), changes[0].path)
            ws.commit(message='Commit message')
            self.assertEqual(list(ws.iter_changes()), [])

    def test_dirty(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        self.build_tree(['subpath'])
        self.assertRaises(WorkspaceDirty, Workspace(tree, use_inotify=self._use_inotify).__enter__)

    def test_reset(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        with Workspace(tree, use_inotify=self._use_inotify) as ws:
            self.build_tree(['blah'])
            ws.reset()
            self.assertPathDoesNotExist('blah')

    def test_tree_path(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        tree.mkdir('subdir')
        tree.commit('Add subdir')
        with Workspace(tree, use_inotify=self._use_inotify) as ws:
            self.assertEqual('foo', ws.tree_path('foo'))
            self.assertEqual('', ws.tree_path())
        with Workspace(tree, subpath='subdir', use_inotify=self._use_inotify) as ws:
            self.assertEqual('subdir/foo', ws.tree_path('foo'))
            self.assertEqual('subdir/', ws.tree_path())

    def test_abspath(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        tree.mkdir('subdir')
        tree.commit('Add subdir')
        with Workspace(tree, use_inotify=self._use_inotify) as ws:
            self.assertEqual(tree.abspath('foo'), ws.abspath('foo'))
            self.assertEqual(tree.abspath(''), ws.abspath())
        with Workspace(tree, subpath='subdir', use_inotify=self._use_inotify) as ws:
            self.assertEqual(tree.abspath('subdir/foo'), ws.abspath('foo'))
            self.assertEqual(tree.abspath('subdir') + '/', ws.abspath(''))
            self.assertEqual(tree.abspath('subdir') + '/', ws.abspath())

    def test_open_containing(self):
        tree = self.make_branch_and_tree('.', format=self._format)
        tree.mkdir('subdir')
        tree.commit('Add subdir')
        ws = Workspace.from_path('subdir')
        self.assertEqual(ws.tree.abspath('.'), tree.abspath('.'))
        self.assertEqual(ws.subpath, 'subdir')
        self.assertEqual(None, ws.use_inotify)