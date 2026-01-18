from lib2to3 import pytree, fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Name, Call, is_tuple, Comma, Attr, ArgList
from libfuturize.fixer_util import touch_import_top
Fixer for 'raise E, V'

From Armin Ronacher's ``python-modernize``.

raise         -> raise
raise E       -> raise E
raise E, 5    -> raise E(5)
raise E, 5, T -> raise E(5).with_traceback(T)
raise E, None, T -> raise E.with_traceback(T)

raise (((E, E'), E''), E'''), 5 -> raise E(5)
raise "foo", V, T               -> warns about string exceptions

raise E, (V1, V2) -> raise E(V1, V2)
raise E, (V1, V2), T -> raise E(V1, V2).with_traceback(T)


CAVEATS:
1) "raise E, V, T" cannot be translated safely in general. If V
   is not a tuple or a (number, string, None) literal, then:

   raise E, V, T -> from future.utils import raise_
                    raise_(E, V, T)
