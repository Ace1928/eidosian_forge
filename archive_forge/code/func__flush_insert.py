from .. import osutils
def _flush_insert(self, start_linenum, end_linenum, new_lines, out_lines, index_lines):
    """Add an 'insert' request to the data stream."""
    bytes_to_insert = b''.join(new_lines[start_linenum:end_linenum])
    insert_length = len(bytes_to_insert)
    for start_byte in range(0, insert_length, 127):
        insert_count = min(insert_length - start_byte, 127)
        out_lines.append(bytes([insert_count]))
        index_lines.append(False)
        insert = bytes_to_insert[start_byte:start_byte + insert_count]
        as_lines = osutils.split_lines(insert)
        out_lines.extend(as_lines)
        index_lines.extend([True] * len(as_lines))