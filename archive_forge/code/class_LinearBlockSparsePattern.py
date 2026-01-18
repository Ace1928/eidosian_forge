import threading
class LinearBlockSparsePattern:
    rlock = threading.RLock()
    row_block_size = 1
    col_block_size = 4
    prev_row_block_size = 1
    prev_col_block_size = 4

    def __init__(self, row_block_size=1, col_block_size=4):
        assert _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size)
        LinearBlockSparsePattern.rlock.acquire()
        LinearBlockSparsePattern.prev_row_block_size = LinearBlockSparsePattern.row_block_size
        LinearBlockSparsePattern.prev_col_block_size = LinearBlockSparsePattern.col_block_size
        LinearBlockSparsePattern.row_block_size = row_block_size
        LinearBlockSparsePattern.col_block_size = col_block_size

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, backtrace):
        LinearBlockSparsePattern.row_block_size = LinearBlockSparsePattern.prev_row_block_size
        LinearBlockSparsePattern.col_block_size = LinearBlockSparsePattern.prev_col_block_size
        LinearBlockSparsePattern.rlock.release()

    @staticmethod
    def block_size():
        return (LinearBlockSparsePattern.row_block_size, LinearBlockSparsePattern.col_block_size)