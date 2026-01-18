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
def _update_backend(cls, backend):
    if cls.__original_docstring__ is None:
        cls.__original_docstring__ = cls.__doc__
    all_keywords = set()
    element_keywords = cls._element_keywords(backend)
    for element, keywords in element_keywords.items():
        with param.logging_level('CRITICAL'):
            all_keywords |= set(keywords)
            setattr(cls, element, cls._create_builder(element, keywords))
    filtered_keywords = [k for k in all_keywords if k not in cls._no_completion]
    sorted_kw_set = sorted(set(filtered_keywords))
    from inspect import Parameter, Signature
    signature = Signature([Parameter('args', Parameter.VAR_POSITIONAL)] + [Parameter(kw, Parameter.KEYWORD_ONLY) for kw in sorted_kw_set])
    cls.__init__.__signature__ = signature