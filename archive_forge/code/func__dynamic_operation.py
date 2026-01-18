import inspect
import os
import shutil
import sys
from collections import defaultdict
from inspect import Parameter, Signature
from pathlib import Path
from types import FunctionType
import param
from pyviz_comms import extension as _pyviz_extension
from ..core import (
from ..core.operation import Operation, OperationCallable
from ..core.options import Keywords, Options, options_policy
from ..core.overlay import Overlay
from ..core.util import merge_options_to_dict
from ..operation.element import function
from ..streams import Params, Stream, streams_list_from_dict
from .settings import OutputSettings, list_backends, list_formats
def _dynamic_operation(self, map_obj):
    """
        Generate function to dynamically apply the operation.
        Wraps an existing HoloMap or DynamicMap.
        """

    def resolve(key, kwargs):
        if not isinstance(map_obj, HoloMap):
            return (key, map_obj)
        elif isinstance(map_obj, DynamicMap) and map_obj._posarg_keys and (not key):
            key = tuple((kwargs[k] for k in map_obj._posarg_keys))
        return (key, map_obj[key])

    def apply(element, *key, **kwargs):
        kwargs = dict(util.resolve_dependent_kwargs(self.p.kwargs), **kwargs)
        processed = self._process(element, key, kwargs)
        if self.p.link_dataset and isinstance(element, Dataset) and isinstance(processed, Dataset) and (processed._dataset is None):
            processed._dataset = element.dataset
        return processed

    def dynamic_operation(*key, **kwargs):
        key, obj = resolve(key, kwargs)
        return apply(obj, *key, **kwargs)
    operation = self.p.operation
    op_kwargs = self.p.kwargs
    if not isinstance(operation, Operation):
        operation = function.instance(fn=apply)
        op_kwargs = {'kwargs': op_kwargs}
    return OperationCallable(dynamic_operation, inputs=[map_obj], link_inputs=self.p.link_inputs, operation=operation, operation_kwargs=op_kwargs)