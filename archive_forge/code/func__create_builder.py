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
def _create_builder(cls, element, completions):

    def builder(cls, spec=None, **kws):
        spec = element if spec is None else f'{element}.{spec}'
        prefix = f'In opts.{element}(...), '
        backend = kws.get('backend', None)
        keys = set(kws.keys())
        if backend:
            allowed_kws = cls._element_keywords(backend, elements=[element])[element]
            invalid = keys - set(allowed_kws)
        else:
            mismatched = {}
            all_valid_kws = set()
            for loaded_backend in Store.loaded_backends():
                valid = set(cls._element_keywords(loaded_backend).get(element, []))
                all_valid_kws |= set(valid)
                if keys <= valid:
                    return Options(spec, **kws)
                mismatched[loaded_backend] = list(keys - valid)
            invalid = keys - all_valid_kws
            if mismatched and (not invalid):
                msg = '{prefix}keywords supplied are mixed across backends. Keyword(s) {info}'
                info = ', '.join(('{} are invalid for {}'.format(', '.join((repr(el) for el in v)), k) for k, v in mismatched.items()))
                raise ValueError(msg.format(info=info, prefix=prefix))
            allowed_kws = completions
        reraise = False
        if invalid:
            try:
                cls._options_error(next(iter(invalid)), element, backend, allowed_kws)
            except ValueError as e:
                msg = str(e)[0].lower() + str(e)[1:]
                reraise = True
            if reraise:
                raise ValueError(prefix + msg)
        return Options(spec, **kws)
    filtered_keywords = [k for k in completions if k not in cls._no_completion]
    sorted_kw_set = sorted(set(filtered_keywords))
    signature = Signature([Parameter('spec', Parameter.POSITIONAL_OR_KEYWORD)] + [Parameter(kw, Parameter.KEYWORD_ONLY) for kw in sorted_kw_set])
    builder.__signature__ = signature
    return classmethod(builder)