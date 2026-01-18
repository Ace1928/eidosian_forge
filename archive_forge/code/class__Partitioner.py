import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
class _Partitioner(object):
    """Convert variable-size writes into part-sized writes

    Call write(data) with variable sized data as needed to write all data. Call
    flush() after all data is written.

    This instance will call send_fn(part_data) as needed in part_size pieces,
    except for the final part which may be shorter than part_size. Make sure to
    call flush() to ensure that a short final part results in a final send_fn
    call.

    """

    def __init__(self, part_size, send_fn):
        self.part_size = part_size
        self.send_fn = send_fn
        self._buffer = []
        self._buffer_size = 0

    def write(self, data):
        if data == b'':
            return
        self._buffer.append(data)
        self._buffer_size += len(data)
        while self._buffer_size > self.part_size:
            self._send_part()

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

    def flush(self):
        if self._buffer_size > 0:
            self._send_part()