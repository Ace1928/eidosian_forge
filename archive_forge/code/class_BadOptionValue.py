import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
class BadOptionValue(errors.BzrError):
    _fmt = 'Bad value "%(value)s" for option "%(name)s".'

    def __init__(self, name, value):
        errors.BzrError.__init__(self, name=name, value=value)