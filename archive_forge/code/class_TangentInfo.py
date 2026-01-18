import collections
import contextlib
from tensorflow.python import pywrap_tfe
class TangentInfo(collections.namedtuple('TangentInfo', ['indices', 'tangents'])):
    """Packed forward accumulator state. The return value of `pack_tangents`."""

    def __new__(cls, indices=None, tangents=None):
        if indices is None:
            indices = ()
        if tangents is None:
            tangents = []
        return super(TangentInfo, cls).__new__(cls, indices, tangents)