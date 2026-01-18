from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class NamedConstant(_Constant):
    """
    L{NamedConstant} defines an attribute to be a named constant within a
    collection defined by a L{Names} subclass.

    L{NamedConstant} is only for use in the definition of L{Names}
    subclasses.  Do not instantiate L{NamedConstant} elsewhere and do not
    subclass it.
    """