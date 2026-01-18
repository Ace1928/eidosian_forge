import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _count_blocks(obj):
    """Count the blocks in an object.

    Splits the data into blocks either on lines or <=64-byte chunks of lines.

    Args:
      obj: The object to count blocks for.

    Returns:
      A dict of block hashcode -> total bytes occurring.
    """
    block_counts: Dict[int, int] = defaultdict(int)
    block = BytesIO()
    n = 0
    block_write = block.write
    block_seek = block.seek
    block_truncate = block.truncate
    block_getvalue = block.getvalue
    for c in chain.from_iterable(obj.as_raw_chunks()):
        c = c.to_bytes(1, 'big')
        block_write(c)
        n += 1
        if c == b'\n' or n == _BLOCK_SIZE:
            value = block_getvalue()
            block_counts[hash(value)] += len(value)
            block_seek(0)
            block_truncate()
            n = 0
    if n > 0:
        last_block = block_getvalue()
        block_counts[hash(last_block)] += len(last_block)
    return block_counts