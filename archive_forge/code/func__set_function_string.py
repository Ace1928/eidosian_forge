from ... import logging
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list
from ...utils.functions import getsource, create_function_from_source
def _set_function_string(self, obj, name, old, new):
    if name == 'function_str':
        if hasattr(new, '__call__'):
            function_source = getsource(new)
            fninfo = new.__code__
        elif isinstance(new, (str, bytes)):
            function_source = new
            fninfo = create_function_from_source(new, self.imports).__code__
        self.inputs.trait_set(trait_change_notify=False, **{'%s' % name: function_source})
        input_names = fninfo.co_varnames[:fninfo.co_argcount]
        new_names = set(input_names) - set(self._input_names)
        add_traits(self.inputs, list(new_names))
        self._input_names.extend(new_names)