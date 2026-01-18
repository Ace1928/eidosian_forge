from .core import Adapter, AdaptationError, Pass
from .lib import int_to_bin, bin_to_int, swap_bytes
from .lib import FlagsContainer, HexString
from .lib.py3compat import BytesIO, decodebytes
class TunnelAdapter(Adapter):
    """
    Adapter for tunneling (as in protocol tunneling). A tunnel is construct
    nested upon another (layering). For parsing, the lower layer first parses
    the data (note: it must return a string!), then the upper layer is called
    to parse that data (bottom-up). For building it works in a top-down manner;
    first the upper layer builds the data, then the lower layer takes it and
    writes it to the stream.

    Parameters:
    * subcon - the lower layer subcon
    * inner_subcon - the upper layer (tunneled/nested) subcon

    Example:
    # a pascal string containing compressed data (zlib encoding), so first
    # the string is read, decompressed, and finally re-parsed as an array
    # of UBInt16
    TunnelAdapter(
        PascalString("data", encoding = "zlib"),
        GreedyRange(UBInt16("elements"))
    )
    """
    __slots__ = ['inner_subcon']

    def __init__(self, subcon, inner_subcon):
        Adapter.__init__(self, subcon)
        self.inner_subcon = inner_subcon

    def _decode(self, obj, context):
        return self.inner_subcon._parse(BytesIO(obj), context)

    def _encode(self, obj, context):
        stream = BytesIO()
        self.inner_subcon._build(obj, stream, context)
        return stream.getvalue()