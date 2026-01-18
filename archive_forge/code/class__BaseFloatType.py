import struct
from llvmlite.ir._utils import _StrCaching
class _BaseFloatType(Type):

    def __new__(cls):
        return cls._instance_cache

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self))

    @classmethod
    def _create_instance(cls):
        cls._instance_cache = super(_BaseFloatType, cls).__new__(cls)