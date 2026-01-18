import types
import weakref
import six
from apitools.base.protorpclite import util
class InvalidDefaultError(FieldDefinitionError):
    """Invalid default provided to field."""