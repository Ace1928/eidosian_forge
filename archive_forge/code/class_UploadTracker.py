import contextlib
from datetime import datetime
import sys
import time
class UploadTracker:
    """Tracker for uploader progress and status."""
    _SUPPORTED_VERBISITY_VALUES = (0, 1)

    def __init__(self, verbosity, one_shot=False):
        if verbosity not in self._SUPPORTED_VERBISITY_VALUES:
            raise ValueError('Unsupported verbosity value %s (supported values: %s)' % (verbosity, self._SUPPORTED_VERBISITY_VALUES))
        self._verbosity = verbosity
        self._stats = UploadStats()
        self._send_count = 0
        self._one_shot = one_shot

    def _dummy_generator(self):
        while True:
            yield 0

    def _overwrite_line_message(self, message, color_code=_STYLE_GREEN):
        """Overwrite the current line with a stylized message."""
        if not self._verbosity:
            return
        message += '.' * 3
        sys.stdout.write(_STYLE_ERASE_LINE + color_code + message + _STYLE_RESET + '\r')
        sys.stdout.flush()

    def _single_line_message(self, message):
        """Write a timestamped single line, with newline, to stdout."""
        if not self._verbosity:
            return
        start_message = '%s[%s]%s %s\n' % (_STYLE_BOLD, readable_time_string(), _STYLE_RESET, message)
        sys.stdout.write(start_message)
        sys.stdout.flush()

    def has_data(self):
        """Determine if any data has been uploaded under the tracker's watch."""
        return self._stats.has_data()

    def _update_cumulative_status(self):
        """Write an update summarizing the data uploaded since the start."""
        if not self._verbosity:
            return
        if not self._stats.has_new_data_since_last_summarize():
            return
        uploaded_str, skipped_str = self._stats.summarize()
        uploaded_message = '%s[%s]%s Total uploaded: %s\n' % (_STYLE_BOLD, readable_time_string(), _STYLE_RESET, uploaded_str)
        sys.stdout.write(uploaded_message)
        if skipped_str:
            sys.stdout.write('%sTotal skipped: %s\n%s' % (_STYLE_DARKGRAY, skipped_str, _STYLE_RESET))
        sys.stdout.flush()

    def add_plugin_name(self, plugin_name):
        self._stats.add_plugin(plugin_name)

    @contextlib.contextmanager
    def send_tracker(self):
        """Create a context manager for a round of data sending."""
        self._send_count += 1
        if self._send_count == 1:
            self._single_line_message('Started scanning logdir.')
        try:
            self._overwrite_line_message('Data upload starting')
            yield
        finally:
            self._update_cumulative_status()
            if self._one_shot:
                self._single_line_message('Done scanning logdir.')
            else:
                self._overwrite_line_message('Listening for new data in logdir', color_code=_STYLE_YELLOW)

    @contextlib.contextmanager
    def scalars_tracker(self, num_scalars):
        """Create a context manager for tracking a scalar batch upload.

        Args:
          num_scalars: Number of scalars in the batch.
        """
        self._overwrite_line_message('Uploading %d scalars' % num_scalars)
        try:
            yield
        finally:
            self._stats.add_scalars(num_scalars)

    @contextlib.contextmanager
    def tensors_tracker(self, num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped):
        """Create a context manager for tracking a tensor batch upload.

        Args:
          num_tensors: Total number of tensors in the batch.
          num_tensors_skipped: Number of tensors skipped (a subset of
            `num_tensors`). Hence this must be `<= num_tensors`.
          tensor_bytes: Total byte size of the tensors in the batch.
          tensor_bytes_skipped: Byte size of skipped tensors in the batch (a
            subset of `tensor_bytes`). Must be `<= tensor_bytes`.
        """
        if num_tensors_skipped:
            message = 'Uploading %d tensors (%s) (Skipping %d tensors, %s)' % (num_tensors - num_tensors_skipped, readable_bytes_string(tensor_bytes - tensor_bytes_skipped), num_tensors_skipped, readable_bytes_string(tensor_bytes_skipped))
        else:
            message = 'Uploading %d tensors (%s)' % (num_tensors, readable_bytes_string(tensor_bytes))
        self._overwrite_line_message(message)
        try:
            yield
        finally:
            self._stats.add_tensors(num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped)

    @contextlib.contextmanager
    def blob_tracker(self, blob_bytes):
        """Creates context manager tracker for uploading a blob.

        Args:
          blob_bytes: Total byte size of the blob being uploaded.
        """
        self._overwrite_line_message('Uploading binary object (%s)' % readable_bytes_string(blob_bytes))
        try:
            yield _BlobTracker(self._stats, blob_bytes)
        finally:
            pass