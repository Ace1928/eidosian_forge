import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
@classmethod
def _adapt_string_for_cast(cls, type_):
    type_ = sqltypes.to_instance(type_)
    if isinstance(type_, sqltypes.CHAR):
        return type_
    elif isinstance(type_, _StringType):
        return CHAR(length=type_.length, charset=type_.charset, collation=type_.collation, ascii=type_.ascii, binary=type_.binary, unicode=type_.unicode, national=False)
    else:
        return CHAR(length=type_.length)