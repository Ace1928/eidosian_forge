import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
class TestConflicts(tests.TestCaseWithTransport):

    def test_resolve_conflict_dir(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('hello', b'hello world4'), ('hello.THIS', b'hello world2'), ('hello.BASE', b'hello world1')])
        os.mkdir('hello.OTHER')
        tree.add('hello', ids=b'q')
        l = bzr_conflicts.ConflictList([bzr_conflicts.TextConflict('hello')])
        l.remove_files(tree)

    def test_select_conflicts(self):
        tree = self.make_branch_and_tree('.')
        clist = bzr_conflicts.ConflictList

        def check_select(not_selected, selected, paths, **kwargs):
            self.assertEqual((not_selected, selected), tree_conflicts.select_conflicts(tree, paths, **kwargs))
        foo = bzr_conflicts.ContentsConflict('foo')
        bar = bzr_conflicts.ContentsConflict('bar')
        tree_conflicts = clist([foo, bar])
        check_select(clist([bar]), clist([foo]), ['foo'])
        check_select(clist(), tree_conflicts, [''], ignore_misses=True, recurse=True)
        foobaz = bzr_conflicts.ContentsConflict('foo/baz')
        tree_conflicts = clist([foobaz, bar])
        check_select(clist([bar]), clist([foobaz]), ['foo'], ignore_misses=True, recurse=True)
        qux = bzr_conflicts.PathConflict('qux', 'foo/baz')
        tree_conflicts = clist([qux])
        check_select(clist(), tree_conflicts, ['foo'], ignore_misses=True, recurse=True)
        check_select(tree_conflicts, clist(), ['foo'], ignore_misses=True)

    def test_resolve_conflicts_recursive(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['dir/', 'dir/hello'])
        tree.add(['dir', 'dir/hello'])
        dirhello = [bzr_conflicts.TextConflict('dir/hello')]
        tree.set_conflicts(dirhello)
        resolve(tree, ['dir'], recursive=False, ignore_misses=True)
        self.assertEqual(dirhello, tree.conflicts())
        resolve(tree, ['dir'], recursive=True, ignore_misses=True)
        self.assertEqual(bzr_conflicts.ConflictList([]), tree.conflicts())