import xcffib
import struct
import io
from . import xproto
from . import render
class ProviderCapability:
    SourceOutput = 1 << 0
    SinkOutput = 1 << 1
    SourceOffload = 1 << 2
    SinkOffload = 1 << 3