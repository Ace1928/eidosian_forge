import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
@inference_state_method_cache()
def _iter_pytest_modules(module_context, skip_own_module=False):
    if not skip_own_module:
        yield module_context
    file_io = module_context.get_value().file_io
    if file_io is not None:
        folder = file_io.get_parent_folder()
        sys_path = module_context.inference_state.get_sys_path()
        last_folder = None
        while any((folder.path.startswith(p) for p in sys_path)):
            file_io = folder.get_file_io('conftest.py')
            if Path(file_io.path) != module_context.py__file__():
                try:
                    m = load_module_from_path(module_context.inference_state, file_io)
                    yield m.as_context()
                except FileNotFoundError:
                    pass
            folder = folder.get_parent_folder()
            if last_folder is not None and folder.path == last_folder.path:
                break
            last_folder = folder
    for names in _PYTEST_FIXTURE_MODULES + _find_pytest_plugin_modules():
        for module_value in module_context.inference_state.import_module(names):
            yield module_value.as_context()