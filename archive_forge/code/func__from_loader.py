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
def _from_loader(loader, string):
    try:
        is_package_method = loader.is_package
    except AttributeError:
        is_package = False
    else:
        is_package = is_package_method(string)
    try:
        get_filename = loader.get_filename
    except AttributeError:
        return (None, is_package)
    else:
        module_path = get_filename(string)
    try:
        f = type(loader).get_source
    except AttributeError:
        raise ImportError('get_source was not defined on loader')
    if f is not importlib.machinery.SourceFileLoader.get_source:
        code = loader.get_source(string)
    else:
        code = _get_source(loader, string)
    if code is None:
        return (None, is_package)
    if isinstance(loader, zipimporter):
        return (ZipFileIO(module_path, code, Path(loader.archive)), is_package)
    return (KnownContentFileIO(module_path, code), is_package)