import struct
from llvmlite.ir._utils import _StrCaching
@classmethod
def _create_instance(cls):
    cls._instance_cache = super(_BaseFloatType, cls).__new__(cls)