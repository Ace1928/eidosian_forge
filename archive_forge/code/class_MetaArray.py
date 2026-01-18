from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
class MetaArray(Subconstruct):
    """
    An array (repeater) of a meta-count. The array will iterate exactly
    `countfunc()` times. Will raise ArrayError if less elements are found.
    See also Array, Range and RepeatUntil.

    Parameters:
    * countfunc - a function that takes the context as a parameter and returns
      the number of elements of the array (count)
    * subcon - the subcon to repeat `countfunc()` times

    Example:
    MetaArray(lambda ctx: 5, UBInt8("foo"))
    """
    __slots__ = ['countfunc']

    def __init__(self, countfunc, subcon):
        Subconstruct.__init__(self, subcon)
        self.countfunc = countfunc
        self._clear_flag(self.FLAG_COPY_CONTEXT)
        self._set_flag(self.FLAG_DYNAMIC)

    def _parse(self, stream, context):
        obj = ListContainer()
        c = 0
        count = self.countfunc(context)
        try:
            if self.subcon.conflags & self.FLAG_COPY_CONTEXT:
                while c < count:
                    obj.append(self.subcon._parse(stream, context.__copy__()))
                    c += 1
            else:
                while c < count:
                    obj.append(self.subcon._parse(stream, context))
                    c += 1
        except ConstructError as ex:
            raise ArrayError('expected %d, found %d' % (count, c), ex)
        return obj

    def _build(self, obj, stream, context):
        count = self.countfunc(context)
        if len(obj) != count:
            raise ArrayError('expected %d, found %d' % (count, len(obj)))
        if self.subcon.conflags & self.FLAG_COPY_CONTEXT:
            for subobj in obj:
                self.subcon._build(subobj, stream, context.__copy__())
        else:
            for subobj in obj:
                self.subcon._build(subobj, stream, context)

    def _sizeof(self, context):
        return self.subcon._sizeof(context) * self.countfunc(context)