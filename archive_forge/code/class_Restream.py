from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Restream(Subconstruct):
    """
    Wraps the stream with a read-wrapper (for parsing) or a
    write-wrapper (for building). The stream wrapper can buffer the data
    internally, reading it from- or writing it to the underlying stream
    as needed. For example, BitStreamReader reads whole bytes from the
    underlying stream, but returns them as individual bits.
    See also Bitwise.

    When the parsing or building is done, the stream's close method
    will be invoked. It can perform any finalization needed for the stream
    wrapper, but it must not close the underlying stream.

    Note:
    * Do not use pointers inside Restream

    Parameters:
    * subcon - the subcon
    * stream_reader - the read-wrapper
    * stream_writer - the write wrapper
    * resizer - a function that takes the size of the subcon and "adjusts"
      or "resizes" it according to the encoding/decoding process.

    Example:
    Restream(BitField("foo", 16),
        stream_reader = BitStreamReader,
        stream_writer = BitStreamWriter,
        resizer = lambda size: size / 8,
    )
    """
    __slots__ = ['stream_reader', 'stream_writer', 'resizer']

    def __init__(self, subcon, stream_reader, stream_writer, resizer):
        Subconstruct.__init__(self, subcon)
        self.stream_reader = stream_reader
        self.stream_writer = stream_writer
        self.resizer = resizer

    def _parse(self, stream, context):
        stream2 = self.stream_reader(stream)
        obj = self.subcon._parse(stream2, context)
        stream2.close()
        return obj

    def _build(self, obj, stream, context):
        stream2 = self.stream_writer(stream)
        self.subcon._build(obj, stream2, context)
        stream2.close()

    def _sizeof(self, context):
        return self.resizer(self.subcon._sizeof(context))