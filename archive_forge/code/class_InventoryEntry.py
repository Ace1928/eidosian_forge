from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class InventoryEntry:
    """Description of a versioned file.

    An InventoryEntry has the following fields, which are also
    present in the XML inventory-entry element:

    file_id

    name
        (within the parent directory)

    parent_id
        file_id of the parent directory, or ROOT_ID

    revision
        the revision_id in which this variation of this file was
        introduced.

    executable
        Indicates that this file should be executable on systems
        that support it.

    text_sha1
        sha-1 of the text of the file

    text_size
        size in bytes of the text of the file

    (reading a version 4 tree created a text_id field.)

    >>> i = Inventory()
    >>> i.path2id('')
    b'TREE_ROOT'
    >>> i.add(InventoryDirectory(b'123', 'src', ROOT_ID))
    InventoryDirectory(b'123', 'src', parent_id=b'TREE_ROOT', revision=None)
    >>> i.add(InventoryFile(b'2323', 'hello.c', parent_id=b'123'))
    InventoryFile(b'2323', 'hello.c', parent_id=b'123', sha1=None, len=None, revision=None)
    >>> shouldbe = {0: '', 1: 'src', 2: 'src/hello.c'}
    >>> for ix, j in enumerate(i.iter_entries()):
    ...   print(j[0] == shouldbe[ix], j[1])
    ...
    True InventoryDirectory(b'TREE_ROOT', '', parent_id=None, revision=None)
    True InventoryDirectory(b'123', 'src', parent_id=b'TREE_ROOT', revision=None)
    True InventoryFile(b'2323', 'hello.c', parent_id=b'123', sha1=None, len=None, revision=None)
    >>> i.add(InventoryFile(b'2324', 'bye.c', b'123'))
    InventoryFile(b'2324', 'bye.c', parent_id=b'123', sha1=None, len=None, revision=None)
    >>> i.add(InventoryDirectory(b'2325', 'wibble', b'123'))
    InventoryDirectory(b'2325', 'wibble', parent_id=b'123', revision=None)
    >>> i.path2id('src/wibble')
    b'2325'
    >>> i.add(InventoryFile(b'2326', 'wibble.c', b'2325'))
    InventoryFile(b'2326', 'wibble.c', parent_id=b'2325', sha1=None, len=None, revision=None)
    >>> i.get_entry(b'2326')
    InventoryFile(b'2326', 'wibble.c', parent_id=b'2325', sha1=None, len=None, revision=None)
    >>> for path, entry in i.iter_entries():
    ...     print(path)
    ...
    <BLANKLINE>
    src
    src/bye.c
    src/hello.c
    src/wibble
    src/wibble/wibble.c
    >>> i.id2path(b'2326')
    'src/wibble/wibble.c'
    """
    RENAMED = 'renamed'
    MODIFIED_AND_RENAMED = 'modified and renamed'
    __slots__ = ['file_id', 'revision', 'parent_id', 'name']
    executable = False
    text_sha1 = None
    text_size = None
    text_id = None
    symlink_target = None
    reference_revision = None

    def detect_changes(self, old_entry):
        """Return a (text_modified, meta_modified) from this to old_entry.

        _read_tree_state must have been called on self and old_entry prior to
        calling detect_changes.
        """
        return (False, False)

    def _diff(self, text_diff, from_label, tree, to_label, to_entry, to_tree, output_to, reverse=False):
        """Perform a diff between two entries of the same kind."""

    def parent_candidates(self, previous_inventories):
        """Find possible per-file graph parents.

        This is currently defined by:
         - Select the last changed revision in the parent inventory.
         - Do deal with a short lived bug in bzr 0.8's development two entries
           that have the same last changed but different 'x' bit settings are
           changed in-place.
        """
        candidates = {}
        for inv in previous_inventories:
            try:
                ie = inv.get_entry(self.file_id)
            except errors.NoSuchId:
                pass
            else:
                if ie.revision in candidates:
                    try:
                        if candidates[ie.revision].executable != ie.executable:
                            candidates[ie.revision].executable = False
                            ie.executable = False
                    except AttributeError:
                        pass
                else:
                    candidates[ie.revision] = ie
        return candidates

    def has_text(self):
        """Return true if the object this entry represents has textual data.

        Note that textual data includes binary content.

        Also note that all entries get weave files created for them.
        This attribute is primarily used when upgrading from old trees that
        did not have the weave index for all inventory entries.
        """
        return False

    def __init__(self, file_id, name, parent_id):
        """Create an InventoryEntry

        The filename must be a single component, relative to the
        parent directory; it cannot be a whole path or relative name.

        >>> e = InventoryFile(b'123', 'hello.c', ROOT_ID)
        >>> e.name
        'hello.c'
        >>> e.file_id
        b'123'
        >>> e = InventoryFile(b'123', 'src/hello.c', ROOT_ID)
        Traceback (most recent call last):
        breezy.bzr.inventory.InvalidEntryName: Invalid entry name: src/hello.c
        """
        if '/' in name:
            raise InvalidEntryName(name=name)
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        self.file_id = file_id
        self.revision = None
        self.name = name
        self.parent_id = parent_id

    def kind_character(self):
        """Return a short kind indicator useful for appending to names."""
        raise errors.BzrError('unknown kind %r' % self.kind)
    known_kinds = ('file', 'directory', 'symlink')

    @staticmethod
    def versionable_kind(kind):
        return kind in ('file', 'directory', 'symlink', 'tree-reference')

    def check(self, checker, rev_id, inv):
        """Check this inventory entry is intact.

        This is a template method, override _check for kind specific
        tests.

        :param checker: Check object providing context for the checks;
             can be used to find out what parts of the repository have already
             been checked.
        :param rev_id: Revision id from which this InventoryEntry was loaded.
             Not necessarily the last-changed revision for this file.
        :param inv: Inventory from which the entry was loaded.
        """
        if self.parent_id is not None:
            if not inv.has_id(self.parent_id):
                raise errors.BzrCheckError('missing parent {{{}}} in inventory for revision {{{}}}'.format(self.parent_id, rev_id))
        checker._add_entry_to_text_key_references(inv, self)
        self._check(checker, rev_id)

    def _check(self, checker, rev_id):
        """Check this inventory entry for kind specific errors."""
        checker._report_items.append('unknown entry kind {!r} in revision {{{}}}'.format(self.kind, rev_id))

    def copy(self):
        """Clone this inventory entry."""
        raise NotImplementedError

    @staticmethod
    def describe_change(old_entry, new_entry):
        """Describe the change between old_entry and this.

        This smells of being an InterInventoryEntry situation, but as its
        the first one, we're making it a static method for now.

        An entry with a different parent, or different name is considered
        to be renamed. Reparenting is an internal detail.
        Note that renaming the parent does not trigger a rename for the
        child entry itself.
        """
        if old_entry is new_entry:
            return 'unchanged'
        elif old_entry is None:
            return 'added'
        elif new_entry is None:
            return 'removed'
        if old_entry.kind != new_entry.kind:
            return 'modified'
        text_modified, meta_modified = new_entry.detect_changes(old_entry)
        if text_modified or meta_modified:
            modified = True
        else:
            modified = False
        if old_entry.parent_id != new_entry.parent_id:
            renamed = True
        elif old_entry.name != new_entry.name:
            renamed = True
        else:
            renamed = False
        if renamed and (not modified):
            return InventoryEntry.RENAMED
        if modified and (not renamed):
            return 'modified'
        if modified and renamed:
            return InventoryEntry.MODIFIED_AND_RENAMED
        return 'unchanged'

    def __repr__(self):
        return '%s(%r, %r, parent_id=%r, revision=%r)' % (self.__class__.__name__, self.file_id, self.name, self.parent_id, self.revision)

    def is_unmodified(self, other):
        other_revision = getattr(other, 'revision', None)
        if other_revision is None:
            return False
        return self.revision == other.revision

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, InventoryEntry):
            return NotImplemented
        return self.file_id == other.file_id and self.name == other.name and (other.symlink_target == self.symlink_target) and (self.text_sha1 == other.text_sha1) and (self.text_size == other.text_size) and (self.text_id == other.text_id) and (self.parent_id == other.parent_id) and (self.kind == other.kind) and (self.revision == other.revision) and (self.executable == other.executable) and (self.reference_revision == other.reference_revision)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        raise ValueError('not hashable')

    def _unchanged(self, previous_ie):
        """Has this entry changed relative to previous_ie.

        This method should be overridden in child classes.
        """
        compatible = True
        if previous_ie.parent_id != self.parent_id:
            compatible = False
        elif previous_ie.name != self.name:
            compatible = False
        elif previous_ie.kind != self.kind:
            compatible = False
        return compatible

    def _read_tree_state(self, path, work_tree):
        """Populate fields in the inventory entry from the given tree.

        Note that this should be modified to be a noop on virtual trees
        as all entries created there are prepopulated.
        """
        pass

    def _forget_tree_state(self):
        pass