from typing import TYPE_CHECKING
from .base import BaseOptions, BaseType
from .field import Field
from .utils import yank_fields_from_attrs
class InterfaceOptions(BaseOptions):
    fields = None
    interfaces = ()