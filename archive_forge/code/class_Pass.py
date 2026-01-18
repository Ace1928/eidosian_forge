from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class Pass(Construct):
    """
    A do-nothing construct, useful as the default case for Switch, or
    to indicate Enums.
    See also Switch and Enum.

    Notes:
    * this construct is a singleton. do not try to instatiate it, as it
      will not work...

    Example:
    Pass
    """
    __slots__ = []

    def _parse(self, stream, context):
        pass

    def _build(self, obj, stream, context):
        assert obj is None

    def _sizeof(self, context):
        return 0

    def __reduce__(self):
        return self.__class__.__name__