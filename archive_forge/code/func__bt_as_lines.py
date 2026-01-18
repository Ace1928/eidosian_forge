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
def _bt_as_lines(bt):
    """
    Converts a backtrace into a list of lines, squashes it a bit on the way.
    """
    return [y for y in itertools.chain(*[x.split('\n') for x in bt]) if y]