import xcffib
import struct
import io
from . import xproto
from . import randr
from . import xfixes
from . import sync
class QueryCapabilitiesCookie(xcffib.Cookie):
    reply_type = QueryCapabilitiesReply