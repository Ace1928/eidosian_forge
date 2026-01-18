import os
import shutil
import sys
from pathlib import Path
from subprocess import check_output
from typing import List, Text, Union
from ..schema import SPEC_VERSION
from ..types import (
class PythonModuleSpec(SpecBase):
    """Helper for a python-based language server spec in the notebook server
    environment
    """
    python_module = ''

    def is_installed(self, mgr: LanguageServerManagerAPI) -> bool:
        spec = self.solve()
        if not spec:
            return False
        if not spec.origin:
            return False
        return True

    def solve(self):
        return __import__('importlib').util.find_spec(self.python_module)

    def __call__(self, mgr: LanguageServerManagerAPI) -> KeyedLanguageServerSpecs:
        is_installed = self.is_installed(mgr)
        return {self.key: {'argv': [sys.executable, '-m', self.python_module, *self.args] if is_installed else [], 'languages': self.languages, 'version': SPEC_VERSION, **self.spec}}