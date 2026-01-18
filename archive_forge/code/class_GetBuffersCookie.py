import xcffib
import struct
import io
from . import xproto
class GetBuffersCookie(xcffib.Cookie):
    reply_type = GetBuffersReply