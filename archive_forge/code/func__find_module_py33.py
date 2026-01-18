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
def _find_module_py33(string, path=None, loader=None, full_name=None, is_global_search=True):
    if not loader:
        spec = importlib.machinery.PathFinder.find_spec(string, path)
        if spec is not None:
            loader = spec.loader
    if loader is None and path is None:
        try:
            spec = importlib.util.find_spec(string)
            if spec is not None:
                loader = spec.loader
        except ValueError as e:
            raise ImportError('Originally  ' + repr(e))
    if loader is None:
        raise ImportError("Couldn't find a loader for {}".format(string))
    return _from_loader(loader, string)