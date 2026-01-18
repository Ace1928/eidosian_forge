import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def _check_changes(self, changes, expected_added=[], expected_removed=[], expected_modified=[], expected_renamed=[], expected_kind_changed=[]):
    """Check the changes in a TreeDelta

        This method checks that the TreeDelta contains the expected
        modifications between the two trees that were used to generate
        it. The required changes are passed in as a list, where
        each entry contains the needed information about the change.

        If you do not wish to assert anything about a particular
        category then pass None instead.

        changes: The TreeDelta to check.
        expected_added: a list of (filename,) tuples that must have
            been added in the delta.
        expected_removed: a list of (filename,) tuples that must have
            been removed in the delta.
        expected_modified: a list of (filename,) tuples that must have
            been modified in the delta.
        expected_renamed: a list of (old_path, new_path) tuples that
            must have been renamed in the delta.
        expected_kind_changed: a list of (path, old_kind, new_kind) tuples
            that must have been changed in the delta.
        """
    renamed = changes.renamed
    added = changes.added + changes.copied
    removed = changes.removed
    modified = changes.modified
    kind_changed = changes.kind_changed
    if expected_renamed is not None:
        self.assertEqual(len(renamed), len(expected_renamed), '{} is renamed, expected {}'.format(renamed, expected_renamed))
        renamed_files = [(item.path[0], item.path[1]) for item in renamed]
        for expected_renamed_entry in expected_renamed:
            expected_renamed_entry = (expected_renamed_entry[0].decode('utf-8'), expected_renamed_entry[1].decode('utf-8'))
            self.assertTrue(expected_renamed_entry in renamed_files, '{} is not renamed, {} are'.format(expected_renamed_entry, renamed_files))
    if expected_added is not None:
        self.assertEqual(len(added), len(expected_added), '%s is added' % str(added))
        added_files = [(item.path[1],) for item in added]
        for expected_added_entry in expected_added:
            expected_added_entry = (expected_added_entry[0].decode('utf-8'),)
            self.assertTrue(expected_added_entry in added_files, '{} is not added, {} are'.format(expected_added_entry, added_files))
    if expected_removed is not None:
        self.assertEqual(len(removed), len(expected_removed), '%s is removed' % str(removed))
        removed_files = [(item.path[0],) for item in removed]
        for expected_removed_entry in expected_removed:
            expected_removed_entry = (expected_removed_entry[0].decode('utf-8'),)
            self.assertTrue(expected_removed_entry in removed_files, '{} is not removed, {} are'.format(expected_removed_entry, removed_files))
    if expected_modified is not None:
        self.assertEqual(len(modified), len(expected_modified), '%s is modified' % str(modified))
        modified_files = [(item.path[1],) for item in modified]
        for expected_modified_entry in expected_modified:
            expected_modified_entry = (expected_modified_entry[0].decode('utf-8'),)
            self.assertTrue(expected_modified_entry in modified_files, '{} is not modified, {} are'.format(expected_modified_entry, modified_files))
    if expected_kind_changed is not None:
        self.assertEqual(len(kind_changed), len(expected_kind_changed), '{} is kind-changed, expected {}'.format(kind_changed, expected_kind_changed))
        kind_changed_files = [(item.path[1], item.kind[0], item.kind[1]) for item in kind_changed]
        for expected_kind_changed_entry in expected_kind_changed:
            expected_kind_changed_entry = (expected_kind_changed_entry[0].decode('utf-8'),) + expected_kind_changed_entry[1:]
            self.assertTrue(expected_kind_changed_entry in kind_changed_files, '{} is not kind-changed, {} are'.format(expected_kind_changed_entry, kind_changed_files))