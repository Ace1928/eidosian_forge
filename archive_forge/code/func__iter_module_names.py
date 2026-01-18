import sys
import os
import inspect
import importlib
from pathlib import Path
from zipfile import ZipFile
from zipimport import zipimporter, ZipImportError
from importlib.machinery import all_suffixes
from jedi.inference.compiled import access
from jedi import debug
from jedi import parser_utils
from jedi.file_io import KnownContentFileIO, ZipFileIO
def _iter_module_names(inference_state, paths):
    for path in paths:
        try:
            dir_entries = ((entry.name, entry.is_dir()) for entry in os.scandir(path))
        except OSError:
            try:
                zip_import_info = zipimporter(path)
                dir_entries = _zip_list_subdirectory(zip_import_info.archive, zip_import_info.prefix)
            except ZipImportError:
                debug.warning('Not possible to list directory: %s', path)
                continue
        for name, is_dir in dir_entries:
            if is_dir:
                if name != '__pycache__' and name.isidentifier():
                    yield name
            else:
                if name.endswith('.pyi'):
                    modname = name[:-4]
                else:
                    modname = inspect.getmodulename(name)
                if modname and '.' not in modname:
                    if modname != '__init__':
                        yield modname