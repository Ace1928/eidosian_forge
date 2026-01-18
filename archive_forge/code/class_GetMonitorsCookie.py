import xcffib
import struct
import io
from . import xproto
from . import render
class GetMonitorsCookie(xcffib.Cookie):
    reply_type = GetMonitorsReply