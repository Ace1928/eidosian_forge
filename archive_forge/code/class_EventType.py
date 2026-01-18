import xcffib
import struct
import io
from . import xproto
class EventType:
    ExchangeComplete = 1
    BlitComplete = 2
    FlipComplete = 3