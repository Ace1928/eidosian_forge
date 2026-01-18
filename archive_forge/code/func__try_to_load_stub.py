import os
import re
from functools import wraps
from collections import namedtuple
from typing import Dict, Mapping, Tuple
from pathlib import Path
from jedi import settings
from jedi.file_io import FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.stub_value import TypingModuleWrapper, StubModuleValue
from jedi.inference.value import ModuleValue
def _try_to_load_stub(inference_state, import_names, python_value_set, parent_module_value, sys_path):
    """
    Trying to load a stub for a set of import_names.

    This is modelled to work like "PEP 561 -- Distributing and Packaging Type
    Information", see https://www.python.org/dev/peps/pep-0561.
    """
    if parent_module_value is None and len(import_names) > 1:
        try:
            parent_module_value = try_to_load_stub_cached(inference_state, import_names[:-1], NO_VALUES, parent_module_value=None, sys_path=sys_path)
        except KeyError:
            pass
    if len(import_names) == 1:
        for p in sys_path:
            init = os.path.join(p, *import_names) + '-stubs' + os.path.sep + '__init__.pyi'
            m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(init), import_names=import_names)
            if m is not None:
                return m
        if import_names[0] == 'django' and python_value_set:
            return _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(str(DJANGO_INIT_PATH)), import_names=import_names)
    for c in python_value_set:
        try:
            method = c.py__file__
        except AttributeError:
            pass
        else:
            file_path = method()
            file_paths = []
            if c.is_namespace():
                file_paths = [os.path.join(p, '__init__.pyi') for p in c.py__path__()]
            elif file_path is not None and file_path.suffix == '.py':
                file_paths = [str(file_path) + 'i']
            for file_path in file_paths:
                m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(file_path), import_names=import_names)
                if m is not None:
                    return m
    m = _load_from_typeshed(inference_state, python_value_set, parent_module_value, import_names)
    if m is not None:
        return m
    if not python_value_set:
        if parent_module_value is not None:
            check_path = parent_module_value.py__path__() or []
            names_for_path = (import_names[-1],)
        else:
            check_path = sys_path
            names_for_path = import_names
        for p in check_path:
            m = _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(os.path.join(p, *names_for_path) + '.pyi'), import_names=import_names)
            if m is not None:
                return m
    return None