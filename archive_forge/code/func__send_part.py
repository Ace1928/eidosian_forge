import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def _send_part(self):
    data = b''.join(self._buffer)
    if len(data) > self.part_size:
        self._buffer = [data[self.part_size:]]
        self._buffer_size = len(self._buffer[0])
    else:
        self._buffer = []
        self._buffer_size = 0
    part = data[:self.part_size]
    self.send_fn(part)