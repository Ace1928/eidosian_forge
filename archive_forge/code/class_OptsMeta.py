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
class OptsMeta(param.parameterized.ParameterizedMetaclass):
    """
    Improve error message when running something
    like: 'hv.opts.Curve()' without a plotting backend.
    """

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            msg = f"No entry for {attr!r} registered; this name may not refer to a valid object or you may need to run 'hv.extension' to select a plotting backend."
            raise AttributeError(msg) from None