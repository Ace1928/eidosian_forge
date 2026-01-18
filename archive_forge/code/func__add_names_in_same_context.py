import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def _add_names_in_same_context(context, string_name):
    if context.tree_node is None:
        return
    until_position = None
    while True:
        filter_ = ParserTreeFilter(parent_context=context, until_position=until_position)
        names = set(filter_.get(string_name))
        if not names:
            break
        yield from names
        ordered = sorted(names, key=lambda x: x.start_pos)
        until_position = ordered[0].start_pos