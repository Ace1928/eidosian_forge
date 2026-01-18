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
class ExternalFunction(Function):
    """
    A named native function (resolvable by LLVM) accepting an explicit
    signature. For internal use only.
    """

    def __init__(self, symbol, sig):
        from numba.core import typing
        self.symbol = symbol
        self.sig = sig
        template = typing.make_concrete_template(symbol, symbol, [sig])
        super(ExternalFunction, self).__init__(template)

    @property
    def key(self):
        return (self.symbol, self.sig)