import os
import subprocess
import winreg
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
from itertools import count
Concrete class that implements an interface to Microsoft Visual C++,
       as defined by the CCompiler abstract class.