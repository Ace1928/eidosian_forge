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
def _load_from_typeshed(inference_state, python_value_set, parent_module_value, import_names):
    import_name = import_names[-1]
    map_ = None
    if len(import_names) == 1:
        map_ = _cache_stub_file_map(inference_state.grammar.version_info)
        import_name = _IMPORT_MAP.get(import_name, import_name)
    elif isinstance(parent_module_value, ModuleValue):
        if not parent_module_value.is_package():
            return None
        paths = parent_module_value.py__path__()
        map_ = _merge_create_stub_map([PathInfo(p, is_third_party=False) for p in paths])
    if map_ is not None:
        path_info = map_.get(import_name)
        if path_info is not None and (not path_info.is_third_party or python_value_set):
            return _try_to_load_stub_from_file(inference_state, python_value_set, file_io=FileIO(path_info.path), import_names=import_names)