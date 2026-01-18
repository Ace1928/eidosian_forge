from typing import TYPE_CHECKING
from ..utils.deprecated import warn_deprecation
from ..utils.get_unbound_function import get_unbound_function
from ..utils.props import props
from .field import Field
from .objecttype import ObjectType, ObjectTypeOptions
from .utils import yank_fields_from_attrs
from .interface import Interface
class MutationOptions(ObjectTypeOptions):
    arguments = None
    output = None
    resolver = None
    interfaces = ()