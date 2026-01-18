import collections
import gzip
import io
import logging
import struct
import numpy as np
def _batched_generator(fin, count, batch_size=1000000.0):
    """Read `count` floats from `fin`.

    Batches up read calls to avoid I/O overhead.  Keeps no more than batch_size
    floats in memory at once.

    Yields floats.

    """
    while count > batch_size:
        batch = _struct_unpack(fin, '@%df' % batch_size)
        for f in batch:
            yield f
        count -= batch_size
    batch = _struct_unpack(fin, '@%df' % count)
    for f in batch:
        yield f