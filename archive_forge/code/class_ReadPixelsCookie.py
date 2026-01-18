import xcffib
import struct
import io
from . import xproto
class ReadPixelsCookie(xcffib.Cookie):
    reply_type = ReadPixelsReply