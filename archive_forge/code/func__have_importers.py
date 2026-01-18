imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
def _have_importers():
    has_py_importer = False
    has_pyx_importer = False
    for importer in sys.meta_path:
        if isinstance(importer, PyxImportMetaFinder):
            if isinstance(importer, PyImportMetaFinder):
                has_py_importer = True
            else:
                has_pyx_importer = True
    return (has_py_importer, has_pyx_importer)