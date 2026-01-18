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
def _is_django_path(directory):
    """ Detects the path of the very well known Django library (if used) """
    try:
        with open(directory.joinpath('manage.py'), 'rb') as f:
            return b'DJANGO_SETTINGS_MODULE' in f.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False