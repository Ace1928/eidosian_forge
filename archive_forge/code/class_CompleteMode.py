import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class CompleteMode:
    Copy = 0
    Flip = 1
    Skip = 2
    SuboptimalCopy = 3