from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _parse_entry(path, file_id, parent_id, last_modified, content):
    entry_factory = {b'dir': _dir_to_entry, b'file': _file_to_entry, b'link': _link_to_entry, b'tree': _tree_to_entry}
    kind = content[0]
    if path.startswith('/'):
        raise AssertionError
    name = basename(path)
    return entry_factory[content[0]](content, name, parent_id, file_id, last_modified)