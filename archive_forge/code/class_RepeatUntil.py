from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class RepeatUntil(Subconstruct):
    """
    An array that repeats until the predicate indicates it to stop. Note that
    the last element (which caused the repeat to exit) is included in the
    return value.

    Parameters:
    * predicate - a predicate function that takes (obj, context) and returns
      True if the stop-condition is met, or False to continue.
    * subcon - the subcon to repeat.

    Example:
    # will read chars until b\x00 (inclusive)
    RepeatUntil(lambda obj, ctx: obj == b"\x00",
        Field("chars", 1)
    )
    """
    __slots__ = ['predicate']

    def __init__(self, predicate, subcon):
        Subconstruct.__init__(self, subcon)
        self.predicate = predicate
        self._clear_flag(self.FLAG_COPY_CONTEXT)
        self._set_flag(self.FLAG_DYNAMIC)

    def _parse(self, stream, context):
        obj = []
        try:
            if self.subcon.conflags & self.FLAG_COPY_CONTEXT:
                while True:
                    subobj = self.subcon._parse(stream, context.__copy__())
                    obj.append(subobj)
                    if self.predicate(subobj, context):
                        break
            else:
                while True:
                    subobj = self.subcon._parse(stream, context)
                    obj.append(subobj)
                    if self.predicate(subobj, context):
                        break
        except ConstructError as ex:
            raise ArrayError('missing terminator', ex)
        return obj

    def _build(self, obj, stream, context):
        terminated = False
        if self.subcon.conflags & self.FLAG_COPY_CONTEXT:
            for subobj in obj:
                self.subcon._build(subobj, stream, context.__copy__())
                if self.predicate(subobj, context):
                    terminated = True
                    break
        else:
            for subobj in obj:
                subobj = bchr(subobj)
                self.subcon._build(subobj, stream, context.__copy__())
                if self.predicate(subobj, context):
                    terminated = True
                    break
        if not terminated:
            raise ArrayError('missing terminator')

    def _sizeof(self, context):
        raise SizeofError("can't calculate size")