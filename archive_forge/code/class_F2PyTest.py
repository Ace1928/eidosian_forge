import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
from pathlib import Path
from numpy.compat import asstr
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
import os
import sys
class F2PyTest:
    code = None
    sources = None
    options = []
    skip = []
    only = []
    suffix = '.f'
    module = None

    @property
    def module_name(self):
        cls = type(self)
        return f'_{cls.__module__.rsplit('.', 1)[-1]}_{cls.__name__}_ext_module'

    def setup_method(self):
        if sys.platform == 'win32':
            pytest.skip('Fails with MinGW64 Gfortran (Issue #9673)')
        if self.module is not None:
            return
        if not has_c_compiler():
            pytest.skip('No C compiler available')
        codes = []
        if self.sources:
            codes.extend(self.sources)
        if self.code is not None:
            codes.append(self.suffix)
        needs_f77 = False
        needs_f90 = False
        needs_pyf = False
        for fn in codes:
            if str(fn).endswith('.f'):
                needs_f77 = True
            elif str(fn).endswith('.f90'):
                needs_f90 = True
            elif str(fn).endswith('.pyf'):
                needs_pyf = True
        if needs_f77 and (not has_f77_compiler()):
            pytest.skip('No Fortran 77 compiler available')
        if needs_f90 and (not has_f90_compiler()):
            pytest.skip('No Fortran 90 compiler available')
        if needs_pyf and (not (has_f90_compiler() or has_f77_compiler())):
            pytest.skip('No Fortran compiler available')
        if self.code is not None:
            self.module = build_code(self.code, options=self.options, skip=self.skip, only=self.only, suffix=self.suffix, module_name=self.module_name)
        if self.sources is not None:
            self.module = build_module(self.sources, options=self.options, skip=self.skip, only=self.only, module_name=self.module_name)