from .. import osutils
class LinesDeltaIndex:
    """This class indexes matches between strings.

    :ivar lines: The 'static' lines that will be preserved between runs.
    :ivar _matching_lines: A dict of {line:[matching offsets]}
    :ivar line_offsets: The byte offset for the end of each line, used to
        quickly map between a matching line number and the byte location
    :ivar endpoint: The total number of bytes in self.line_offsets
    """
    _MIN_MATCH_BYTES = 10
    _SOFT_MIN_MATCH_BYTES = 200

    def __init__(self, lines):
        self.lines = []
        self.line_offsets = []
        self.endpoint = 0
        self._matching_lines = {}
        self.extend_lines(lines, [True] * len(lines))

    def _update_matching_lines(self, new_lines, index):
        matches = self._matching_lines
        start_idx = len(self.lines)
        if len(new_lines) != len(index):
            raise AssertionError("The number of lines to be indexed does not match the index/don't index flags: %d != %d" % (len(new_lines), len(index)))
        for idx, do_index in enumerate(index):
            if not do_index:
                continue
            line = new_lines[idx]
            try:
                matches[line].add(start_idx + idx)
            except KeyError:
                matches[line] = {start_idx + idx}

    def get_matches(self, line):
        """Return the lines which match the line in right."""
        try:
            return self._matching_lines[line]
        except KeyError:
            return None

    def _get_longest_match(self, lines, pos):
        """Look at all matches for the current line, return the longest.

        :param lines: The lines we are matching against
        :param pos: The current location we care about
        :param locations: A list of lines that matched the current location.
            This may be None, but often we'll have already found matches for
            this line.
        :return: (start_in_self, start_in_lines, num_lines)
            All values are the offset in the list (aka the line number)
            If start_in_self is None, then we have no matches, and this line
            should be inserted in the target.
        """
        range_start = pos
        range_len = 0
        prev_locations = None
        max_pos = len(lines)
        matching = self._matching_lines
        while pos < max_pos:
            try:
                locations = matching[lines[pos]]
            except KeyError:
                pos += 1
                break
            if prev_locations is None:
                prev_locations = locations
                range_len = 1
                locations = None
            else:
                next_locations = locations.intersection([loc + 1 for loc in prev_locations])
                if next_locations:
                    prev_locations = set(next_locations)
                    range_len += 1
                    locations = None
                else:
                    break
            pos += 1
        if prev_locations is None:
            return (None, pos)
        smallest = min(prev_locations)
        return ((smallest - range_len + 1, range_start, range_len), pos)

    def get_matching_blocks(self, lines, soft=False):
        """Return the ranges in lines which match self.lines.

        :param lines: lines to compress
        :return: A list of (old_start, new_start, length) tuples which reflect
            a region in self.lines that is present in lines.  The last element
            of the list is always (old_len, new_len, 0) to provide a end point
            for generating instructions from the matching blocks list.
        """
        result = []
        pos = 0
        max_pos = len(lines)
        result_append = result.append
        min_match_bytes = self._MIN_MATCH_BYTES
        if soft:
            min_match_bytes = self._SOFT_MIN_MATCH_BYTES
        while pos < max_pos:
            block, pos = self._get_longest_match(lines, pos)
            if block is not None:
                if block[-1] < min_match_bytes:
                    old_start, new_start, range_len = block
                    matched_bytes = sum(map(len, lines[new_start:new_start + range_len]))
                    if matched_bytes < min_match_bytes:
                        block = None
            if block is not None:
                result_append(block)
        result_append((len(self.lines), len(lines), 0))
        return result

    def extend_lines(self, lines, index):
        """Add more lines to the left-lines list.

        :param lines: A list of lines to add
        :param index: A True/False for each node to define if it should be
            indexed.
        """
        self._update_matching_lines(lines, index)
        self.lines.extend(lines)
        endpoint = self.endpoint
        for line in lines:
            endpoint += len(line)
            self.line_offsets.append(endpoint)
        if len(self.line_offsets) != len(self.lines):
            raise AssertionError('Somehow the line offset indicator got out of sync with the line counter.')
        self.endpoint = endpoint

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

    def _flush_copy(self, old_start_linenum, num_lines, out_lines, index_lines):
        if old_start_linenum == 0:
            first_byte = 0
        else:
            first_byte = self.line_offsets[old_start_linenum - 1]
        stop_byte = self.line_offsets[old_start_linenum + num_lines - 1]
        num_bytes = stop_byte - first_byte
        for start_byte in range(first_byte, stop_byte, 64 * 1024):
            num_bytes = min(64 * 1024, stop_byte - start_byte)
            copy_bytes = encode_copy_instruction(start_byte, num_bytes)
            out_lines.append(copy_bytes)
            index_lines.append(False)

    def make_delta(self, new_lines, bytes_length, soft=False):
        """Compute the delta for this content versus the original content."""
        out_lines = [b'', b'', encode_base128_int(bytes_length)]
        index_lines = [False, False, False]
        output_handler = _OutputHandler(out_lines, index_lines, self._MIN_MATCH_BYTES)
        blocks = self.get_matching_blocks(new_lines, soft=soft)
        current_line_num = 0
        for old_start, new_start, range_len in blocks:
            if new_start != current_line_num:
                output_handler.add_insert(new_lines[current_line_num:new_start])
            current_line_num = new_start + range_len
            if range_len:
                if old_start == 0:
                    first_byte = 0
                else:
                    first_byte = self.line_offsets[old_start - 1]
                last_byte = self.line_offsets[old_start + range_len - 1]
                output_handler.add_copy(first_byte, last_byte)
        return (out_lines, index_lines)