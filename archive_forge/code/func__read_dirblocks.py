import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def _read_dirblocks(state):
    """Read in the dirblocks for the given DirState object.

    This is tightly bound to the DirState internal representation. It should be
    thought of as a member function, which is only separated out so that we can
    re-write it in pyrex.

    :param state: A DirState object.
    :return: None
    """
    state._state_file.seek(state._end_of_header)
    text = state._state_file.read()
    fields = text.split(b'\x00')
    trailing = fields.pop()
    if trailing != b'':
        raise DirstateCorrupt(state, 'trailing garbage: {!r}'.format(trailing))
    cur = 1
    num_present_parents = state._num_present_parents()
    tree_count = 1 + num_present_parents
    entry_size = state._fields_per_entry()
    expected_field_count = entry_size * state._num_entries
    field_count = len(fields)
    if field_count - cur != expected_field_count:
        raise DirstateCorrupt(state, 'field count incorrect %s != %s, entry_size=%s, num_entries=%s fields=%r' % (field_count - cur, expected_field_count, entry_size, state._num_entries, fields))
    if num_present_parents == 1:
        _int = int
        _iter = iter(fields)
        next = getattr(_iter, '__next__', None)
        if next is None:
            next = _iter.next
        for x in range(cur):
            next()
        state._dirblocks = [(b'', []), (b'', [])]
        current_block = state._dirblocks[0][1]
        current_dirname = b''
        append_entry = current_block.append
        for count in range(state._num_entries):
            dirname = next()
            name = next()
            file_id = next()
            if dirname != current_dirname:
                current_block = []
                current_dirname = dirname
                state._dirblocks.append((current_dirname, current_block))
                append_entry = current_block.append
            entry = ((current_dirname, name, file_id), [(next(), next(), _int(next()), next() == b'y', next()), (next(), next(), _int(next()), next() == b'y', next())])
            trailing = next()
            if trailing != b'\n':
                raise ValueError('trailing garbage in dirstate: %r' % trailing)
            append_entry(entry)
        state._split_root_dirblock_into_contents()
    else:
        fields_to_entry = state._get_fields_to_entry()
        entries = [fields_to_entry(fields[pos:pos + entry_size]) for pos in range(cur, field_count, entry_size)]
        state._entries_to_current_state(entries)
    state._dirblock_state = DirState.IN_MEMORY_UNMODIFIED