from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
class CppCompilerRunner(CompilerRunner):
    compiler_dict = OrderedDict([('gnu', 'g++'), ('intel', 'icpc'), ('llvm', 'clang++')])
    standards = ('c++98', 'c++0x')
    std_formater = {'g++': '-std={}'.format, 'icpc': '-std={}'.format, 'clang++': '-std={}'.format}
    compiler_name_vendor_mapping = {'g++': 'gnu', 'icpc': 'intel', 'clang++': 'llvm'}