from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
class TestUnsupportedViews(TestCaseWithWorkingTree):
    """Formats that don't support views should give reasonable errors."""

    def setUp(self):
        fmt = self.workingtree_format
        supported = getattr(fmt, 'supports_views')
        if supported is None:
            warn("Format %s doesn't declare whether it supports views or not" % fmt)
            raise TestSkipped('No view support at all')
        if supported():
            raise TestSkipped('Format %s declares that views are supported' % fmt)
        super().setUp()

    def test_view_methods_raise(self):
        wt = self.make_branch_and_tree('wt')
        self.assertRaises(_mod_views.ViewsNotSupported, wt.views.set_view_info, 'bar', {'bar': ['bars/']})
        self.assertRaises(_mod_views.ViewsNotSupported, wt.views.get_view_info)
        self.assertRaises(_mod_views.ViewsNotSupported, wt.views.lookup_view, 'foo')
        self.assertRaises(_mod_views.ViewsNotSupported, wt.views.set_view, 'foo', 'bar')
        self.assertRaises(_mod_views.ViewsNotSupported, wt.views.delete_view, 'foo')