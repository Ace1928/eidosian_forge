import hashlib
import os
from typing import Generic, TypeVar, Union, Dict, Optional, Any
from pathlib import Path
from parso._compatibility import is_pypy
from parso.pgen2 import generate_grammar
from parso.utils import split_lines, python_bytes_to_unicode, \
from parso.python.diff import DiffParser
from parso.python.tokenize import tokenize_lines, tokenize
from parso.python.token import PythonTokenTypes
from parso.cache import parser_cache, load_module, try_to_save_module
from parso.parser import BaseParser
from parso.python.parser import Parser as PythonParser
from parso.python.errors import ErrorFinderConfig
from parso.python import pep8
from parso.file_io import FileIO, KnownContentFileIO
from parso.normalizer import RefactoringNormalizer, NormalizerConfig
def _get_token_namespace(self):
    ns = self._token_namespace
    if ns is None:
        raise ValueError('The token namespace should be set.')
    return ns