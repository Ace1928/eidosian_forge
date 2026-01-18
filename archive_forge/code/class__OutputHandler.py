from .. import osutils
class _OutputHandler:
    """A simple class which just tracks how to split up an insert request."""

    def __init__(self, out_lines, index_lines, min_len_to_index):
        self.out_lines = out_lines
        self.index_lines = index_lines
        self.min_len_to_index = min_len_to_index
        self.cur_insert_lines = []
        self.cur_insert_len = 0

    def add_copy(self, start_byte, end_byte):
        for start_byte in range(start_byte, end_byte, 64 * 1024):
            num_bytes = min(64 * 1024, end_byte - start_byte)
            copy_bytes = encode_copy_instruction(start_byte, num_bytes)
            self.out_lines.append(copy_bytes)
            self.index_lines.append(False)

    def _flush_insert(self):
        if not self.cur_insert_lines:
            return
        if self.cur_insert_len > 127:
            raise AssertionError('We cannot insert more than 127 bytes at a time.')
        self.out_lines.append(bytes([self.cur_insert_len]))
        self.index_lines.append(False)
        self.out_lines.extend(self.cur_insert_lines)
        if self.cur_insert_len < self.min_len_to_index:
            self.index_lines.extend([False] * len(self.cur_insert_lines))
        else:
            self.index_lines.extend([True] * len(self.cur_insert_lines))
        self.cur_insert_lines = []
        self.cur_insert_len = 0

    def _insert_long_line(self, line):
        self._flush_insert()
        line_len = len(line)
        for start_index in range(0, line_len, 127):
            next_len = min(127, line_len - start_index)
            self.out_lines.append(bytes([next_len]))
            self.index_lines.append(False)
            self.out_lines.append(line[start_index:start_index + next_len])
            self.index_lines.append(False)

    def add_insert(self, lines):
        if self.cur_insert_lines != []:
            raise AssertionError('self.cur_insert_lines must be empty when adding a new insert')
        for line in lines:
            if len(line) > 127:
                self._insert_long_line(line)
            else:
                next_len = len(line) + self.cur_insert_len
                if next_len > 127:
                    self._flush_insert()
                    self.cur_insert_lines = [line]
                    self.cur_insert_len = len(line)
                else:
                    self.cur_insert_lines.append(line)
                    self.cur_insert_len = next_len
        self._flush_insert()