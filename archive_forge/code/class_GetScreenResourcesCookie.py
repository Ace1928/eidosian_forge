import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenResourcesCookie(xcffib.Cookie):
    reply_type = GetScreenResourcesReply