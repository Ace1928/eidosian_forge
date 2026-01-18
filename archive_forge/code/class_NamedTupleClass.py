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
class NamedTupleClass(Callable, Opaque):
    """
    Type class for namedtuple classes.
    """

    def __init__(self, instance_class):
        self.instance_class = instance_class
        name = 'class(%s)' % instance_class
        super(NamedTupleClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        return None

    def get_call_signatures(self):
        return ((), True)

    def get_impl_key(self, sig):
        return type(self)

    @property
    def key(self):
        return self.instance_class