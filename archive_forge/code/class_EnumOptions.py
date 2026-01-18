from enum import Enum as PyEnum
from graphene.utils.subclass_with_meta import SubclassWithMeta_Meta
from .base import BaseOptions, BaseType
from .unmountedtype import UnmountedType
class EnumOptions(BaseOptions):
    enum = None
    deprecation_reason = None