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
@inference_state_as_method_param_cache()
def _get_sys_path(self, inference_state, add_parent_paths=True, add_init_paths=False):
    """
        Keep this method private for all users of jedi. However internally this
        one is used like a public method.
        """
    suffixed = list(self.added_sys_path)
    prefixed = []
    if self._sys_path is None:
        sys_path = list(self._get_base_sys_path(inference_state))
    else:
        sys_path = list(self._sys_path)
    if self._smart_sys_path:
        prefixed.append(str(self._path))
        if inference_state.script_path is not None:
            suffixed += map(str, discover_buildout_paths(inference_state, inference_state.script_path))
            if add_parent_paths:
                traversed = []
                for parent_path in inference_state.script_path.parents:
                    if parent_path == self._path or self._path not in parent_path.parents:
                        break
                    if not add_init_paths and parent_path.joinpath('__init__.py').is_file():
                        continue
                    traversed.append(str(parent_path))
                suffixed += reversed(traversed)
    if self._django:
        prefixed.append(str(self._path))
    path = prefixed + sys_path + suffixed
    return list(_remove_duplicates_from_path(path))