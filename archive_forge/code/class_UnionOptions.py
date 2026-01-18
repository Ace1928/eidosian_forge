from typing import TYPE_CHECKING
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
class UnionOptions(BaseOptions):
    types = ()