from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
class CCompilerRunner(CompilerRunner):
    compiler_dict = OrderedDict([('gnu', 'gcc'), ('intel', 'icc'), ('llvm', 'clang')])
    standards = ('c89', 'c90', 'c99', 'c11')
    std_formater = {'gcc': '-std={}'.format, 'icc': '-std={}'.format, 'clang': '-std={}'.format}
    compiler_name_vendor_mapping = {'gcc': 'gnu', 'icc': 'intel', 'clang': 'llvm'}