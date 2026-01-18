import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
class TestSmartAddTreeUnicode(per_workingtree.TestCaseWithWorkingTree):
    _test_needs_features = [features.UnicodeFilenameFeature]

    def setUp(self):
        super().setUp()
        self.build_tree(['å'])
        self.wt = self.make_branch_and_tree('.')
        self.overrideAttr(osutils, 'normalized_filename')

    def test_requires_normalized_unicode_filenames_fails_on_unnormalized(self):
        """Adding unnormalized unicode filenames fail if and only if the
        workingtree format has the requires_normalized_unicode_filenames flag
        set and the underlying filesystem doesn't normalize.
        """
        osutils.normalized_filename = osutils._accessible_normalized_filename
        if self.workingtree_format.requires_normalized_unicode_filenames and sys.platform != 'darwin':
            self.assertRaises(transport.NoSuchFile, self.wt.smart_add, ['å'])
        else:
            self.wt.smart_add(['å'])

    def test_accessible_explicit(self):
        osutils.normalized_filename = osutils._accessible_normalized_filename
        if self.workingtree_format.requires_normalized_unicode_filenames:
            raise tests.TestNotApplicable('Working tree format smart_add requires normalized unicode filenames')
        self.wt.smart_add(['å'])
        self.wt.lock_read()
        self.addCleanup(self.wt.unlock)
        self.assertEqual([('', 'directory'), ('å', 'file')], [(path, ie.kind) for path, ie in self.wt.iter_entries_by_dir()])

    def test_accessible_implicit(self):
        osutils.normalized_filename = osutils._accessible_normalized_filename
        if self.workingtree_format.requires_normalized_unicode_filenames:
            raise tests.TestNotApplicable('Working tree format smart_add requires normalized unicode filenames')
        self.wt.smart_add([])
        self.wt.lock_read()
        self.addCleanup(self.wt.unlock)
        self.assertEqual([('', 'directory'), ('å', 'file')], [(path, ie.kind) for path, ie in self.wt.iter_entries_by_dir()])

    def test_inaccessible_explicit(self):
        osutils.normalized_filename = osutils._inaccessible_normalized_filename
        self.assertRaises(errors.InvalidNormalization, self.wt.smart_add, ['å'])

    def test_inaccessible_implicit(self):
        osutils.normalized_filename = osutils._inaccessible_normalized_filename
        self.assertRaises(errors.InvalidNormalization, self.wt.smart_add, [])