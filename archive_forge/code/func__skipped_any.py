import contextlib
from datetime import datetime
import sys
import time
def _skipped_any(self):
    """Whether any data was skipped."""
    return self._num_tensors_skipped or self._num_blobs_skipped