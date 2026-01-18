from typing import TYPE_CHECKING
from .base import BaseOptions, BaseType
from .inputfield import InputField
from .unmountedtype import UnmountedType
from .utils import yank_fields_from_attrs
class InputObjectTypeOptions(BaseOptions):
    fields = None
    container = None