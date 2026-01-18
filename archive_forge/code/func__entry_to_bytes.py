from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _entry_to_bytes(self, entry):
    """Serialise entry as a single bytestring.

        :param Entry: An inventory entry.
        :return: A bytestring for the entry.

        The BNF:
        ENTRY ::= FILE | DIR | SYMLINK | TREE
        FILE ::= "file: " COMMON SEP SHA SEP SIZE SEP EXECUTABLE
        DIR ::= "dir: " COMMON
        SYMLINK ::= "symlink: " COMMON SEP TARGET_UTF8
        TREE ::= "tree: " COMMON REFERENCE_REVISION
        COMMON ::= FILE_ID SEP PARENT_ID SEP NAME_UTF8 SEP REVISION
        SEP ::= "
"
        """
    if entry.parent_id is not None:
        parent_str = entry.parent_id
    else:
        parent_str = b''
    name_str = entry.name.encode('utf8')
    if entry.kind == 'file':
        if entry.executable:
            exec_str = b'Y'
        else:
            exec_str = b'N'
        return b'file: %s\n%s\n%s\n%s\n%s\n%d\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.text_sha1, entry.text_size, exec_str)
    elif entry.kind == 'directory':
        return b'dir: %s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision)
    elif entry.kind == 'symlink':
        return b'symlink: %s\n%s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.symlink_target.encode('utf8'))
    elif entry.kind == 'tree-reference':
        return b'tree: %s\n%s\n%s\n%s\n%s' % (entry.file_id, parent_str, name_str, entry.revision, entry.reference_revision)
    else:
        raise ValueError('unknown kind %r' % entry.kind)