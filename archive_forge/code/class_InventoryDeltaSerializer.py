from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
class InventoryDeltaSerializer:
    """Serialize inventory deltas."""

    def __init__(self, versioned_root, tree_references):
        """Create an InventoryDeltaSerializer.

        :param versioned_root: If True, any root entry that is seen is expected
            to be versioned, and root entries can have any fileid.
        :param tree_references: If True support tree-reference entries.
        """
        self._versioned_root = versioned_root
        self._tree_references = tree_references
        self._entry_to_content = {'directory': _directory_content, 'file': _file_content, 'symlink': _link_content}
        if tree_references:
            self._entry_to_content['tree-reference'] = _reference_content

    def delta_to_lines(self, old_name, new_name, delta_to_new):
        """Return a line sequence for delta_to_new.

        Both the versioned_root and tree_references flags must be set via
        require_flags before calling this.

        :param old_name: A UTF8 revision id for the old inventory.  May be
            NULL_REVISION if there is no older inventory and delta_to_new
            includes the entire inventory contents.
        :param new_name: The version name of the inventory we create with this
            delta.
        :param delta_to_new: An inventory delta such as Inventory.apply_delta
            takes.
        :return: The serialized delta as lines.
        """
        if not isinstance(old_name, bytes):
            raise TypeError('old_name should be str, got {!r}'.format(old_name))
        if not isinstance(new_name, bytes):
            raise TypeError('new_name should be str, got {!r}'.format(new_name))
        lines = [b'', b'', b'', b'', b'']
        to_line = self._delta_item_to_line
        for delta_item in delta_to_new:
            line = to_line(delta_item, new_name)
            if line.__class__ != bytes:
                raise InventoryDeltaError('to_line gave non-bytes output %(line)r', line=lines[-1])
            lines.append(line)
        lines.sort()
        lines[0] = b'format: %s\n' % FORMAT_1
        lines[1] = b'parent: %s\n' % old_name
        lines[2] = b'version: %s\n' % new_name
        lines[3] = b'versioned_root: %s\n' % self._serialize_bool(self._versioned_root)
        lines[4] = b'tree_references: %s\n' % self._serialize_bool(self._tree_references)
        return lines

    def _serialize_bool(self, value):
        if value:
            return b'true'
        else:
            return b'false'

    def _delta_item_to_line(self, delta_item, new_version):
        """Convert delta_item to a line."""
        oldpath, newpath, file_id, entry = delta_item
        if newpath is None:
            oldpath_utf8 = b'/' + oldpath.encode('utf8')
            newpath_utf8 = b'None'
            parent_id = b''
            last_modified = NULL_REVISION
            content = b'deleted\x00\x00'
        else:
            if oldpath is None:
                oldpath_utf8 = b'None'
            else:
                oldpath_utf8 = b'/' + oldpath.encode('utf8')
            if newpath == '/':
                raise AssertionError("Bad inventory delta: '/' is not a valid newpath (should be '') in delta item %r" % (delta_item,))
            newpath_utf8 = b'/' + newpath.encode('utf8')
            parent_id = entry.parent_id or b''
            last_modified = entry.revision
            if newpath_utf8 == b'/' and (not self._versioned_root):
                if last_modified != new_version:
                    raise InventoryDeltaError('Version present for / in %(fileid)r (%(last)r != %(new)r)', fileid=file_id, last=last_modified, new=new_version)
            if last_modified is None:
                raise InventoryDeltaError('no version for fileid %(fileid)r', fileid=file_id)
            content = self._entry_to_content[entry.kind](entry)
        return b'%s\x00%s\x00%s\x00%s\x00%s\x00%s\n' % (oldpath_utf8, newpath_utf8, file_id, parent_id, last_modified, content)