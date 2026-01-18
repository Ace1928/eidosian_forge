import threading
from tensorboard._vendor.bleach.sanitizer import Cleaner
import markdown
from tensorboard import context as _context
from tensorboard.backend import experiment_id as _experiment_id
from tensorboard.util import tb_logging
class _MetadataVersionChecker:
    """TensorBoard-internal utility for warning when data is too new.

    Specify a maximum known `version` number as stored in summary
    metadata, and automatically reject and warn on data from newer
    versions. This keeps a (single) bit of internal state to handle
    logging a warning to the user at most once.

    This should only be used by plugins bundled with TensorBoard, since
    it may instruct users to upgrade their copy of TensorBoard.
    """

    def __init__(self, data_kind, latest_known_version):
        """Initialize a `_MetadataVersionChecker`.

        Args:
          data_kind: A human-readable description of the kind of data
            being read, like "scalar" or "histogram" or "PR curve".
          latest_known_version: Highest tolerated value of `version`,
            like `0`.
        """
        self._data_kind = data_kind
        self._latest_known_version = latest_known_version
        self._warned = False

    def ok(self, version, run, tag):
        """Test whether `version` is permitted, else complain."""
        if 0 <= version <= self._latest_known_version:
            return True
        self._maybe_warn(version, run, tag)
        return False

    def _maybe_warn(self, version, run, tag):
        if self._warned:
            return
        self._warned = True
        logger.warning('Some %s data is too new to be read by this version of TensorBoard. Upgrading TensorBoard may fix this. (sample: run %r, tag %r, data version %r)', self._data_kind, run, tag, version)