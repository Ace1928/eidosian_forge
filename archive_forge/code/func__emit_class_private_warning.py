from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def _emit_class_private_warning(self, pos, name):
    warning(pos, "Global name %s matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release." % name, 1)