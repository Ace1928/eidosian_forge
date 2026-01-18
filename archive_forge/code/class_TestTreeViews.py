from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
class TestTreeViews(TestCaseWithWorkingTree):

    def setUp(self):
        fmt = self.workingtree_format
        f = getattr(fmt, 'supports_views')
        if f is None:
            raise TestSkipped("format %s doesn't declare whether it supports views, assuming not" % fmt)
        if not f():
            raise TestNotApplicable("format %s doesn't support views" % fmt)
        super().setUp()

    def test_views_initially_empty(self):
        wt = self.make_branch_and_tree('wt')
        current, views = wt.views.get_view_info()
        self.assertEqual(None, current)
        self.assertEqual({}, views)

    def test_set_and_get_view_info(self):
        wt = self.make_branch_and_tree('wt')
        view_current = 'view-name'
        view_dict = {view_current: ['dir-1'], 'other-name': ['dir-2']}
        wt.views.set_view_info(view_current, view_dict)
        current, views = wt.views.get_view_info()
        self.assertEqual(view_current, current)
        self.assertEqual(view_dict, views)
        wt = WorkingTree.open('wt')
        current, views = wt.views.get_view_info()
        self.assertEqual(view_current, current)
        self.assertEqual(view_dict, views)
        self.assertRaises(_mod_views.NoSuchView, wt.views.set_view_info, 'yet-another', view_dict)
        current, views = wt.views.get_view_info()
        self.assertEqual(view_current, current)
        self.assertEqual(view_dict, views)
        wt.views.set_view_info(None, view_dict)
        current, views = wt.views.get_view_info()
        self.assertEqual(None, current)
        self.assertEqual(view_dict, views)

    def test_lookup_view(self):
        wt = self.make_branch_and_tree('wt')
        view_current = 'view-name'
        view_dict = {view_current: ['dir-1'], 'other-name': ['dir-2']}
        wt.views.set_view_info(view_current, view_dict)
        result = wt.views.lookup_view()
        self.assertEqual(result, ['dir-1'])
        result = wt.views.lookup_view('other-name')
        self.assertEqual(result, ['dir-2'])

    def test_set_view(self):
        wt = self.make_branch_and_tree('wt')
        wt.views.set_view('view-1', ['dir-1'])
        current, views = wt.views.get_view_info()
        self.assertEqual('view-1', current)
        self.assertEqual({'view-1': ['dir-1']}, views)
        wt.views.set_view('view-2', ['dir-2'], make_current=False)
        current, views = wt.views.get_view_info()
        self.assertEqual('view-1', current)
        self.assertEqual({'view-1': ['dir-1'], 'view-2': ['dir-2']}, views)

    def test_unicode_view(self):
        wt = self.make_branch_and_tree('wt')
        view_name = 'ば'
        view_files = ['foo', 'bar/']
        view_dict = {view_name: view_files}
        wt.views.set_view_info(view_name, view_dict)
        current, views = wt.views.get_view_info()
        self.assertEqual(view_name, current)
        self.assertEqual(view_dict, views)

    def test_no_such_view(self):
        wt = self.make_branch_and_tree('wt')
        try:
            wt.views.lookup_view('opaque')
        except _mod_views.NoSuchView as e:
            self.assertEqual(e.view_name, 'opaque')
            self.assertEqual(str(e), 'No such view: opaque.')
        else:
            self.fail("didn't get expected exception")

    def test_delete_view(self):
        wt = self.make_branch_and_tree('wt')
        view_name = 'α'
        view_files = ['alphas/']
        wt.views.set_view(view_name, view_files)
        wt.views.delete_view(view_name)
        self.assertRaises(_mod_views.NoSuchView, wt.views.lookup_view, view_name)
        self.assertEqual(wt.views.get_view_info()[1], {})
        self.assertRaises(_mod_views.NoSuchView, wt.views.delete_view, view_name)
        self.assertRaises(_mod_views.NoSuchView, wt.views.delete_view, view_name + '2')

    def test_check_path_in_view(self):
        wt = self.make_branch_and_tree('wt')
        view_current = 'view-name'
        view_dict = {view_current: ['dir-1'], 'other-name': ['dir-2']}
        wt.views.set_view_info(view_current, view_dict)
        self.assertEqual(_mod_views.check_path_in_view(wt, 'dir-1'), None)
        self.assertEqual(_mod_views.check_path_in_view(wt, 'dir-1/sub'), None)
        self.assertRaises(_mod_views.FileOutsideView, _mod_views.check_path_in_view, wt, 'dir-2')
        self.assertRaises(_mod_views.FileOutsideView, _mod_views.check_path_in_view, wt, 'dir-2/sub')
        self.assertRaises(_mod_views.FileOutsideView, _mod_views.check_path_in_view, wt, 'other')