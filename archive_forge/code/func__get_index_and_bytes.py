from io import BytesIO
from .. import errors, osutils, transport
from ..commands import Command, display_command
from ..option import Option
from ..workingtree import WorkingTree
from . import btree_index, static_tuple
def _get_index_and_bytes(self, trans, basename):
    """Create a BTreeGraphIndex and raw bytes."""
    bt = btree_index.BTreeGraphIndex(trans, basename, None)
    bytes = trans.get_bytes(basename)
    bt._file = BytesIO(bytes)
    bt._size = len(bytes)
    return (bt, bytes)