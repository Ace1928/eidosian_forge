import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class GetClientDisconnectModeCookie(xcffib.Cookie):
    reply_type = GetClientDisconnectModeReply