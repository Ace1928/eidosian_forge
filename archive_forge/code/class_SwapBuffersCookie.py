import xcffib
import struct
import io
from . import xproto
class SwapBuffersCookie(xcffib.Cookie):
    reply_type = SwapBuffersReply