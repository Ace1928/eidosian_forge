import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def _find_global_variables(names, search_name):
    for name in names:
        if name.tree_name is None:
            continue
        module_context = name.get_root_context()
        try:
            method = module_context.get_global_filter
        except AttributeError:
            continue
        else:
            for global_name in method().get(search_name):
                yield global_name
                c = module_context.create_context(global_name.tree_name)
                yield from _add_names_in_same_context(c, global_name.string_name)