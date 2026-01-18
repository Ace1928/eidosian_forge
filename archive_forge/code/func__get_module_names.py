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
def _get_module_names(self, search_path=None, in_module=None):
    """
        Get the names of all modules in the search_path. This means file names
        and not names defined in the files.
        """
    if search_path is None:
        sys_path = self._sys_path_with_modifications(is_completion=True)
    else:
        sys_path = search_path
    return list(iter_module_names(self._inference_state, self._module_context, sys_path, module_cls=ImportName if in_module is None else SubModuleName, add_builtin_modules=search_path is None and in_module is None))