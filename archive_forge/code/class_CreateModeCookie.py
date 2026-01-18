import xcffib
import struct
import io
from . import xproto
from . import render
class CreateModeCookie(xcffib.Cookie):
    reply_type = CreateModeReply