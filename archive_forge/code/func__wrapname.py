import struct
from llvmlite.ir._utils import _StrCaching
def _wrapname(x):
    return '"{0}"'.format(x.replace('\\', '\\5c').replace('"', '\\22'))