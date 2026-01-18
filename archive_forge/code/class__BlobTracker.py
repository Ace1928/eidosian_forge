import contextlib
from datetime import datetime
import sys
import time
class _BlobTracker:

    def __init__(self, upload_stats, blob_bytes):
        self._upload_stats = upload_stats
        self._blob_bytes = blob_bytes

    def mark_uploaded(self, is_uploaded):
        self._upload_stats.add_blob(self._blob_bytes, is_skipped=not is_uploaded)