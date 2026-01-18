import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def _find_python_files_in_sys_path(inference_state, module_contexts):
    sys_path = inference_state.get_sys_path()
    except_paths = set()
    yielded_paths = [m.py__file__() for m in module_contexts]
    for module_context in module_contexts:
        file_io = module_context.get_value().file_io
        if file_io is None:
            continue
        folder_io = file_io.get_parent_folder()
        while True:
            path = folder_io.path
            if not any((path.startswith(p) for p in sys_path)) or path in except_paths:
                break
            for file_io in recurse_find_python_files(folder_io, except_paths):
                if file_io.path not in yielded_paths:
                    yield file_io
            except_paths.add(path)
            folder_io = folder_io.get_parent_folder()