import os
from pathlib import Path
from parso.python import tree
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.file_io import FolderIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference import sys_path
from jedi.inference import helpers
from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.utils import unite
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import ImportName, SubModuleName
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.typeshed import import_module_decorator, \
from jedi.inference.compiled.subprocess.functions import ImplicitNSInfo
from jedi.plugins import plugin_manager
def _load_python_module(inference_state, file_io, import_names=None, is_package=False):
    module_node = inference_state.parse(file_io=file_io, cache=True, diff_cache=settings.fast_parser, cache_path=settings.cache_directory)
    from jedi.inference.value import ModuleValue
    return ModuleValue(inference_state, module_node, file_io=file_io, string_names=import_names, code_lines=get_cached_code_lines(inference_state.grammar, file_io.path), is_package=is_package)