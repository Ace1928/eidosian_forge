import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
class CapturingDisplayHook(object):

    def __init__(self, shell, outputs=None):
        self.shell = shell
        if outputs is None:
            outputs = []
        self.outputs = outputs

    def __call__(self, result=None):
        if result is None:
            return
        format_dict, md_dict = self.shell.display_formatter.format(result)
        self.outputs.append({'data': format_dict, 'metadata': md_dict})