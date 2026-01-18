from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _directory_content(entry):
    """Serialize the content component of entry which is a directory.

    :param entry: An InventoryDirectory.
    """
    return b'dir'