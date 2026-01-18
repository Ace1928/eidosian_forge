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
class ImplicitNSInfo:
    """Stores information returned from an implicit namespace spec"""

    def __init__(self, name, paths):
        self.name = name
        self.paths = paths