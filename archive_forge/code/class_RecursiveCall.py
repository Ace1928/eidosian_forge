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
class RecursiveCall(Opaque):
    """
    Recursive call to a Dispatcher.
    """
    _overloads = None

    def __init__(self, dispatcher_type):
        assert isinstance(dispatcher_type, Dispatcher)
        self.dispatcher_type = dispatcher_type
        name = 'recursive(%s)' % (dispatcher_type,)
        super(RecursiveCall, self).__init__(name)
        if self._overloads is None:
            self._overloads = {}

    def add_overloads(self, args, qualname, uid):
        """Add an overload of the function.

        Parameters
        ----------
        args :
            argument types
        qualname :
            function qualifying name
        uid :
            unique id
        """
        self._overloads[args] = _RecursiveCallOverloads(qualname, uid)

    def get_overloads(self, args):
        """Get the qualifying name and unique id for the overload given the
        argument types.
        """
        return self._overloads[args]

    @property
    def key(self):
        return self.dispatcher_type