from .. import osutils
def add_copy(self, start_byte, end_byte):
    for start_byte in range(start_byte, end_byte, 64 * 1024):
        num_bytes = min(64 * 1024, end_byte - start_byte)
        copy_bytes = encode_copy_instruction(start_byte, num_bytes)
        self.out_lines.append(copy_bytes)
        self.index_lines.append(False)