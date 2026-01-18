import contextlib
from datetime import datetime
import sys
import time
def add_blob(self, blob_bytes, is_skipped):
    """Add a blob.

        Args:
          blob_bytes: Byte size of the blob.
          is_skipped: Whether the uploading of the blob is skipped due to
            reasons such as size exceeding limit.
        """
    self._refresh_last_data_added_timestamp()
    self._num_blobs += 1
    self._blob_bytes += blob_bytes
    if is_skipped:
        self._num_blobs_skipped += 1
        self._blob_bytes_skipped += blob_bytes