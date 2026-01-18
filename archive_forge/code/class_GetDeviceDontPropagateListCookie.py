import xcffib
import struct
import io
from . import xfixes
from . import xproto
class GetDeviceDontPropagateListCookie(xcffib.Cookie):
    reply_type = GetDeviceDontPropagateListReply