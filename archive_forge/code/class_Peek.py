from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Peek(Subconstruct):
    """
    Peeks at the stream: parses without changing the stream position.
    See also Union. If the end of the stream is reached when peeking,
    returns None.

    Notes:
    * requires a seekable stream.

    Parameters:
    * subcon - the subcon to peek at
    * perform_build - whether or not to perform building. by default this
      parameter is set to False, meaning building is a no-op.

    Example:
    Peek(UBInt8("foo"))
    """
    __slots__ = ['perform_build']

    def __init__(self, subcon, perform_build=False):
        Subconstruct.__init__(self, subcon)
        self.perform_build = perform_build

    def _parse(self, stream, context):
        pos = stream.tell()
        try:
            return self.subcon._parse(stream, context)
        except FieldError:
            pass
        finally:
            stream.seek(pos)

    def _build(self, obj, stream, context):
        if self.perform_build:
            self.subcon._build(obj, stream, context)

    def _sizeof(self, context):
        return 0