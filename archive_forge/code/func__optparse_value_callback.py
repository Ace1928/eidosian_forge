import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _optparse_value_callback(self, cb_value):

    def cb(option, opt, value, parser):
        v = self.type(cb_value)
        setattr(parser.values, self._param_name, v)
        if self.custom_callback is not None:
            self.custom_callback(option, self._param_name, v, parser)
    return cb