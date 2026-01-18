import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
class _PickleableWeakRef(weakref.ref):
    """
    Allow a weakref to be pickled.

    Note that if the object referred to is not kept alive elsewhere in the
    pickle, the weakref will immediately expire after being constructed.
    """

    def __getnewargs__(self):
        obj = self()
        if obj is None:
            raise ReferenceError('underlying object has vanished')
        return (obj,)