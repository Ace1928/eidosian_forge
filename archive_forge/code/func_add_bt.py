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
def add_bt(error):
    if isinstance(error, BaseException):
        bt = traceback.format_exception(type(error), error, error.__traceback__)
    else:
        bt = ['']
    nd2indent = '\n{}'.format(2 * indent)
    errstr = _termcolor.reset(nd2indent + nd2indent.join(_bt_as_lines(bt)))
    return _termcolor.reset(errstr)