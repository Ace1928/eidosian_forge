import os
import warnings
from .errors import (
from .ccompiler import CCompiler, gen_preprocess_options
from .file_util import write_file
from ._modified import newer
from ._log import log
Concrete class that implements an interface to the Borland C/C++
    compiler, as defined by the CCompiler abstract class.
    