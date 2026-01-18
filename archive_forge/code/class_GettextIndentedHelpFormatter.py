import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
class GettextIndentedHelpFormatter(optparse.IndentedHelpFormatter):
    """Adds gettext() call to format_option()"""

    def __init__(self):
        optparse.IndentedHelpFormatter.__init__(self)

    def format_option(self, option):
        """code taken from Python's optparse.py"""
        if option.help:
            from .i18n import gettext
            option.help = gettext(option.help)
        return optparse.IndentedHelpFormatter.format_option(self, option)