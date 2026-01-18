import xcffib
import struct
import io
from . import xproto
class ConnectCookie(xcffib.Cookie):
    reply_type = ConnectReply