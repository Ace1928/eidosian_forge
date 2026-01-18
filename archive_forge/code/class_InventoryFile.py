from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class InventoryFile(InventoryEntry):
    """A file in an inventory."""
    __slots__ = ['text_sha1', 'text_size', 'text_id', 'executable']
    kind = 'file'

    def __init__(self, file_id, name, parent_id):
        super().__init__(file_id, name, parent_id)
        self.text_sha1 = None
        self.text_size = None
        self.text_id = None
        self.executable = False

    def _check(self, checker, tree_revision_id):
        """See InventoryEntry._check"""
        checker.add_pending_item(tree_revision_id, ('texts', self.file_id, self.revision), b'text', self.text_sha1)
        if self.text_size is None:
            checker._report_items.append('fileid {{{}}} in {{{}}} has None for text_size'.format(self.file_id, tree_revision_id))

    def copy(self):
        other = InventoryFile(self.file_id, self.name, self.parent_id)
        other.executable = self.executable
        other.text_id = self.text_id
        other.text_sha1 = self.text_sha1
        other.text_size = self.text_size
        other.revision = self.revision
        return other

    def detect_changes(self, old_entry):
        """See InventoryEntry.detect_changes."""
        text_modified = self.text_sha1 != old_entry.text_sha1
        meta_modified = self.executable != old_entry.executable
        return (text_modified, meta_modified)

    def _diff(self, text_diff, from_label, tree, to_label, to_entry, to_tree, output_to, reverse=False):
        """See InventoryEntry._diff."""
        from breezy.diff import DiffText
        from_file_id = self.file_id
        if to_entry:
            to_file_id = to_entry.file_id
            to_path = to_tree.id2path(to_file_id)
        else:
            to_file_id = None
            to_path = None
        if from_file_id is not None:
            from_path = tree.id2path(from_file_id)
        else:
            from_path = None
        if reverse:
            to_file_id, from_file_id = (from_file_id, to_file_id)
            tree, to_tree = (to_tree, tree)
            from_label, to_label = (to_label, from_label)
        differ = DiffText(tree, to_tree, output_to, 'utf-8', '', '', text_diff)
        return differ.diff_text(from_path, to_path, from_label, to_label, from_file_id, to_file_id)

    def has_text(self):
        """See InventoryEntry.has_text."""
        return True

    def kind_character(self):
        """See InventoryEntry.kind_character."""
        return ''

    def _read_tree_state(self, path, work_tree):
        """See InventoryEntry._read_tree_state."""
        self.text_sha1 = work_tree.get_file_sha1(path)
        self.executable = work_tree.is_executable(path)

    def __repr__(self):
        return '%s(%r, %r, parent_id=%r, sha1=%r, len=%s, revision=%s)' % (self.__class__.__name__, self.file_id, self.name, self.parent_id, self.text_sha1, self.text_size, self.revision)

    def _forget_tree_state(self):
        self.text_sha1 = None

    def _unchanged(self, previous_ie):
        """See InventoryEntry._unchanged."""
        compatible = super()._unchanged(previous_ie)
        if self.text_sha1 != previous_ie.text_sha1:
            compatible = False
        else:
            self.text_size = previous_ie.text_size
        if self.executable != previous_ie.executable:
            compatible = False
        return compatible