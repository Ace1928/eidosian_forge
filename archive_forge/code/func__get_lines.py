from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _get_lines(self, file_id):
    """Get the lines for a file-id."""
    return osutils.split_lines(self._get_data(file_id))