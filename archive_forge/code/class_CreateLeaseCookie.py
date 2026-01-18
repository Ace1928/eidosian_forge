import xcffib
import struct
import io
from . import xproto
from . import render
class CreateLeaseCookie(xcffib.Cookie):
    reply_type = CreateLeaseReply