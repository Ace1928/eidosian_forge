from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class IndexingAdapter(Adapter):
    """
    Adapter for indexing a list (getting a single item from that list)

    Parameters:
    * subcon - the subcon to index
    * index - the index of the list to get
    """
    __slots__ = ['index']

    def __init__(self, subcon, index):
        Adapter.__init__(self, subcon)
        if type(index) is not int:
            raise TypeError('index must be an integer', type(index))
        self.index = index

    def _encode(self, obj, context):
        return [None] * self.index + [obj]

    def _decode(self, obj, context):
        return obj[self.index]