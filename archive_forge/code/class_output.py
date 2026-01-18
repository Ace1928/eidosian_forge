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
class output(param.ParameterizedFunction):
    """
    Utility function to set output either at the global level or on a
    specific object.

    To set output globally use:

    output(options)

    Where options may be an options specification string (as accepted by
    the IPython opts magic) or an options specifications dictionary.

    For instance:

    output("backend='bokeh'") # Or equivalently
    output(backend='bokeh')

    To set save output from a specific object do disk using the
    'filename' argument, you can supply the object as the first
    positional argument and supply the filename keyword:

    curve = hv.Curve([1,2,3])
    output(curve, filename='curve.png')

    For compatibility with the output magic, you can supply the object
    as the second argument after the string specification:

    curve = hv.Curve([1,2,3])
    output("filename='curve.png'", curve)

    These two modes are equivalent to the IPython output line magic and
    the cell magic respectively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def info(cls):
        deprecate = ['filename', 'info', 'mode']
        options = Store.output_settings.options
        defaults = Store.output_settings.defaults
        keys = [k for k, v in options.items() if k not in deprecate and v != defaults[k]]
        pairs = {k: options[k] for k in sorted(keys)}
        if 'backend' not in keys:
            pairs['backend'] = Store.current_backend
        if ':' in pairs['backend']:
            pairs['backend'] = pairs['backend'].split(':')[0]
        keywords = ', '.join((f'{k}={pairs[k]!r}' for k in sorted(pairs.keys())))
        print(f'output({keywords})')

    def __call__(self, *args, **options):
        help_prompt = 'For help with hv.util.output call help(hv.util.output)'
        line, obj = (None, None)
        if len(args) > 2:
            raise TypeError('The opts utility accepts one or two positional arguments.')
        if len(args) == 1 and (not isinstance(args[0], str)):
            obj = args[0]
        elif len(args) == 1:
            line = args[0]
        elif len(args) == 2:
            line, obj = args
        if isinstance(obj, Dimensioned):
            if line:
                options = Store.output_settings.extract_keywords(line, {})
            for k in options.keys():
                if k not in Store.output_settings.allowed:
                    raise KeyError(f'Invalid keyword: {k}')

            def display_fn(obj, renderer):
                try:
                    from IPython.display import display
                except ImportError:
                    return
                display(obj)
            Store.output_settings.output(line=line, cell=obj, cell_runner=display_fn, help_prompt=help_prompt, **options)
        elif obj is not None:
            return obj
        else:
            Store.output_settings.output(line=line, help_prompt=help_prompt, **options)