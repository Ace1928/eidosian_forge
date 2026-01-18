import json
from pathlib import Path
from itertools import chain
from jedi import debug
from jedi.api.environment import get_cached_default_environment, create_environment
from jedi.api.exceptions import WrongVersion
from jedi.api.completion import search_in_module
from jedi.api.helpers import split_search_string, get_module_names
from jedi.inference.imports import load_module_from_path, \
from jedi.inference.sys_path import discover_buildout_paths
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.inference.references import recurse_find_python_folders_and_files, search_in_file_ios
from jedi.file_io import FolderIO
def _try_to_skip_duplicates(func):

    def wrapper(*args, **kwargs):
        found_tree_nodes = []
        found_modules = []
        for definition in func(*args, **kwargs):
            tree_node = definition._name.tree_name
            if tree_node is not None and tree_node in found_tree_nodes:
                continue
            if definition.type == 'module' and definition.module_path is not None:
                if definition.module_path in found_modules:
                    continue
                found_modules.append(definition.module_path)
            yield definition
            found_tree_nodes.append(tree_node)
    return wrapper