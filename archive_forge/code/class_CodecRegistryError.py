import codecs
import sys
from . import aliases
class CodecRegistryError(LookupError, SystemError):
    pass