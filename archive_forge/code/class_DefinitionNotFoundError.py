import types
import weakref
import six
from apitools.base.protorpclite import util
class DefinitionNotFoundError(Error):
    """Raised when definition is not found."""