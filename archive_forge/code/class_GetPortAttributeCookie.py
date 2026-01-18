import xcffib
import struct
import io
from . import xproto
from . import shm
class GetPortAttributeCookie(xcffib.Cookie):
    reply_type = GetPortAttributeReply