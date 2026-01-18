from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding, Attr

Fixer for:
it.__next__() -> it.next().
next(it) -> it.next().
