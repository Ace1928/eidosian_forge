from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class PrefilterTransformer(Configurable):
    """Transform a line of user input."""
    priority = Integer(100).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    enabled = Bool(True).tag(config=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        super(PrefilterTransformer, self).__init__(shell=shell, prefilter_manager=prefilter_manager, **kwargs)
        self.prefilter_manager.register_transformer(self)

    def transform(self, line, continue_prompt):
        """Transform a line, returning the new one."""
        return None

    def __repr__(self):
        return '<%s(priority=%r, enabled=%r)>' % (self.__class__.__name__, self.priority, self.enabled)