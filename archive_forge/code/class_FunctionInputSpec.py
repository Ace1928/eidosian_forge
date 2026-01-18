from ... import logging
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list
from ...utils.functions import getsource, create_function_from_source
class FunctionInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    function_str = traits.Str(mandatory=True, desc='code for function')