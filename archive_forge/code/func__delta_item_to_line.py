from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
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