import threading
def _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
    return row_block_size == 1 and col_block_size == 4 or (row_block_size == 8 and col_block_size == 1)