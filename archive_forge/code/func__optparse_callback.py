import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _optparse_callback(self, option, opt, value, parser):
    values = getattr(parser.values, self._param_name)
    if value == '-':
        del values[:]
    else:
        values.append(self.type(value))
    if self.custom_callback is not None:
        self.custom_callback(option, self._param_name, values, parser)