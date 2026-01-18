import types
import weakref
import six
from apitools.base.protorpclite import util
class BooleanField(Field):
    """Field definition for boolean values."""
    VARIANTS = frozenset([Variant.BOOL])
    DEFAULT_VARIANT = Variant.BOOL
    type = bool