import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def _get_paths_from_buildout_script(inference_state, buildout_script_path):
    file_io = FileIO(str(buildout_script_path))
    try:
        module_node = inference_state.parse(file_io=file_io, cache=True, cache_path=settings.cache_directory)
    except IOError:
        debug.warning('Error trying to read buildout_script: %s', buildout_script_path)
        return
    from jedi.inference.value import ModuleValue
    module_context = ModuleValue(inference_state, module_node, file_io=file_io, string_names=None, code_lines=get_cached_code_lines(inference_state.grammar, buildout_script_path)).as_context()
    yield from check_sys_path_modifications(module_context)