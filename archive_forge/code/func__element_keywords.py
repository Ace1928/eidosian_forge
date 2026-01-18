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
@classmethod
def _element_keywords(cls, backend, elements=None):
    """Returns a dictionary of element names to allowed keywords"""
    if backend not in Store.loaded_backends():
        return {}
    mapping = {}
    backend_options = Store.options(backend)
    elements = elements if elements is not None else backend_options.keys()
    for element in elements:
        if '.' in element:
            continue
        element = element if isinstance(element, tuple) else (element,)
        element_keywords = []
        options = backend_options['.'.join(element)]
        for group in Options._option_groups:
            element_keywords.extend(options[group].allowed_keywords)
        mapping[element[0]] = element_keywords
    return mapping