import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
class throw_error:

    def __init__(self, mess):
        self.mess = mess

    def __call__(self, var):
        mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
        raise F2PYError(mess)