import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
class OptionParser(optparse.OptionParser):
    """OptionParser that raises exceptions instead of exiting"""
    DEFAULT_VALUE = object()

    def __init__(self):
        optparse.OptionParser.__init__(self)
        self.formatter = GettextIndentedHelpFormatter()

    def error(self, message):
        raise errors.CommandError(message)