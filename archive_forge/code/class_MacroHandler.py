from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class MacroHandler(PrefilterHandler):
    handler_name = Unicode('macro')

    def handle(self, line_info):
        obj = self.shell.user_ns.get(line_info.ifun)
        pre_space = line_info.pre_whitespace
        line_sep = '\n' + pre_space
        return pre_space + line_sep.join(obj.value.splitlines())