import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def _check_fs(inference_state, file_io, regex):
    try:
        code = file_io.read()
    except FileNotFoundError:
        return None
    code = python_bytes_to_unicode(code, errors='replace')
    if not regex.search(code):
        return None
    new_file_io = KnownContentFileIO(file_io.path, code)
    m = load_module_from_path(inference_state, new_file_io)
    if m.is_compiled():
        return None
    return m.as_context()