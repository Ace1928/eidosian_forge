from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
class InventoryDeltaDeserializer:
    """Deserialize inventory deltas."""

    def __init__(self, allow_versioned_root=True, allow_tree_references=True):
        """Create an InventoryDeltaDeserializer.

        :param versioned_root: If True, any root entry that is seen is expected
            to be versioned, and root entries can have any fileid.
        :param tree_references: If True support tree-reference entries.
        """
        self._allow_versioned_root = allow_versioned_root
        self._allow_tree_references = allow_tree_references

    def _deserialize_bool(self, value):
        if value == b'true':
            return True
        elif value == b'false':
            return False
        else:
            raise InventoryDeltaError('value %(val)r is not a bool', val=value)

    def parse_text_bytes(self, lines):
        """Parse the text bytes of a serialized inventory delta.

        If versioned_root and/or tree_references flags were set via
        require_flags, then the parsed flags must match or a BzrError will be
        raised.

        :param lines: The lines to parse. This can be obtained by calling
            delta_to_lines.
        :return: (parent_id, new_id, versioned_root, tree_references,
            inventory_delta)
        """
        if not lines:
            raise InventoryDeltaError('inventory delta is empty')
        if not lines[-1].endswith(b'\n'):
            raise InventoryDeltaError('last line not empty: %(line)r', line=lines[-1])
        lines = [line.rstrip(b'\n') for line in lines]
        if not lines or lines[0] != b'format: %s' % FORMAT_1:
            raise InventoryDeltaError('unknown format %(line)r', line=lines[0:1])
        if len(lines) < 2 or not lines[1].startswith(b'parent: '):
            raise InventoryDeltaError('missing parent: marker')
        delta_parent_id = lines[1][8:]
        if len(lines) < 3 or not lines[2].startswith(b'version: '):
            raise InventoryDeltaError('missing version: marker')
        delta_version_id = lines[2][9:]
        if len(lines) < 4 or not lines[3].startswith(b'versioned_root: '):
            raise InventoryDeltaError('missing versioned_root: marker')
        delta_versioned_root = self._deserialize_bool(lines[3][16:])
        if len(lines) < 5 or not lines[4].startswith(b'tree_references: '):
            raise InventoryDeltaError('missing tree_references: marker')
        delta_tree_references = self._deserialize_bool(lines[4][17:])
        if not self._allow_versioned_root and delta_versioned_root:
            raise IncompatibleInventoryDelta('versioned_root not allowed')
        result = []
        seen_ids = set()
        line_iter = iter(lines)
        for i in range(5):
            next(line_iter)
        for line in line_iter:
            oldpath_utf8, newpath_utf8, file_id, parent_id, last_modified, content = line.split(b'\x00', 5)
            parent_id = parent_id or None
            if file_id in seen_ids:
                raise InventoryDeltaError('duplicate file id %(fileid)r', fileid=file_id)
            seen_ids.add(file_id)
            if newpath_utf8 == b'/' and (not delta_versioned_root) and (last_modified != delta_version_id):
                raise InventoryDeltaError('Versioned root found: %(line)r', line=line)
            elif newpath_utf8 != b'None' and last_modified[-1:] == b':':
                raise InventoryDeltaError('special revisionid found: %(line)r', line=line)
            if content.startswith(b'tree\x00'):
                if delta_tree_references is False:
                    raise InventoryDeltaError('Tree reference found (but header said tree_references: false): %(line)r', line=line)
                elif not self._allow_tree_references:
                    raise IncompatibleInventoryDelta('Tree reference not allowed')
            if oldpath_utf8 == b'None':
                oldpath = None
            elif oldpath_utf8[:1] != b'/':
                raise InventoryDeltaError('oldpath invalid (does not start with /): %(path)r', path=oldpath_utf8)
            else:
                oldpath_utf8 = oldpath_utf8[1:]
                oldpath = oldpath_utf8.decode('utf8')
            if newpath_utf8 == b'None':
                newpath = None
            elif newpath_utf8[:1] != b'/':
                raise InventoryDeltaError('newpath invalid (does not start with /): %(path)r', path=newpath_utf8)
            else:
                newpath_utf8 = newpath_utf8[1:]
                newpath = newpath_utf8.decode('utf8')
            content_tuple = tuple(content.split(b'\x00'))
            if content_tuple[0] == b'deleted':
                entry = None
            else:
                entry = _parse_entry(newpath, file_id, parent_id, last_modified, content_tuple)
            delta_item = (oldpath, newpath, file_id, entry)
            result.append(delta_item)
        return (delta_parent_id, delta_version_id, delta_versioned_root, delta_tree_references, result)