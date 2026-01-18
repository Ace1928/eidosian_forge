from ..construct import (
class StreamOffset(Construct):
    """
    Captures the current stream offset

    Parameters:
    * name - the name of the value

    Example:
    StreamOffset("item_offset")
    """
    __slots__ = []

    def __init__(self, name):
        Construct.__init__(self, name)
        self._set_flag(self.FLAG_DYNAMIC)

    def _parse(self, stream, context):
        return stream.tell()

    def _build(self, obj, stream, context):
        context[self.name] = stream.tell()

    def _sizeof(self, context):
        return 0