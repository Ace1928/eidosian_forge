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
def _prepare_infer_import(module_context, tree_name):
    import_node = search_ancestor(tree_name, 'import_name', 'import_from')
    import_path = import_node.get_path_for_name(tree_name)
    from_import_name = None
    try:
        from_names = import_node.get_from_names()
    except AttributeError:
        pass
    else:
        if len(from_names) + 1 == len(import_path):
            from_import_name = import_path[-1]
            import_path = from_names
    importer = Importer(module_context.inference_state, tuple(import_path), module_context, import_node.level)
    return (from_import_name, tuple(import_path), import_node.level, importer.follow())