import types
import weakref
import six
from apitools.base.protorpclite import util
class BytesField(Field):
    """Field definition for byte string values."""
    VARIANTS = frozenset([Variant.BYTES])
    DEFAULT_VARIANT = Variant.BYTES
    type = bytes