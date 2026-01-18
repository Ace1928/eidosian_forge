from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def IfThenElse(name, predicate, then_subcon, else_subcon):
    """an if-then-else conditional construct: if the predicate indicates True,
    `then_subcon` will be used; otherwise `else_subcon`
    * name - the name of the construct
    * predicate - a function taking the context as an argument and returning
      True or False
    * then_subcon - the subcon that will be used if the predicate returns True
    * else_subcon - the subcon that will be used if the predicate returns False
    """
    return Switch(name, lambda ctx: bool(predicate(ctx)), {True: then_subcon, False: else_subcon})