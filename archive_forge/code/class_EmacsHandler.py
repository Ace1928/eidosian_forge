from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class EmacsHandler(PrefilterHandler):
    handler_name = Unicode('emacs')
    esc_strings = List([])

    def handle(self, line_info):
        """Handle input lines marked by python-mode."""
        return line_info.line