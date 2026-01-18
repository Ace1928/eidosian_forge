import struct
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
class RecordWriter:
    """Write encoded protobuf to a file with packing defined in tensorflow."""

    def __init__(self, writer):
        """Open a file to keep the tensorboard records.

        Args:
        writer: A file-like object that implements `write`, `flush` and `close`.
        """
        self._writer = writer

    def write(self, data):
        header = struct.pack('<Q', len(data))
        header_crc = struct.pack('<I', masked_crc32c(header))
        footer_crc = struct.pack('<I', masked_crc32c(data))
        self._writer.write(header + header_crc + data + footer_crc)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

    @property
    def closed(self):
        return self._writer.closed