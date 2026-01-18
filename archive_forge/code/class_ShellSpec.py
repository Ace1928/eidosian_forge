import os
import shutil
import sys
from pathlib import Path
from subprocess import check_output
from typing import List, Text, Union
from ..schema import SPEC_VERSION
from ..types import (
class ShellSpec(SpecBase):
    """Helper for a language server spec for executables on $PATH in the
    notebook server environment.
    """
    cmd = ''
    is_installed_args: List[Token] = []

    def is_installed(self, mgr: LanguageServerManagerAPI) -> bool:
        cmd = self.solve()
        if not cmd:
            return False
        if not self.is_installed_args:
            return bool(cmd)
        else:
            check_result = check_output([cmd, *self.is_installed_args]).decode(encoding='utf-8')
            return check_result != ''

    def solve(self) -> Union[str, None]:
        for ext in ['', '.cmd', '.bat', '.exe']:
            cmd = shutil.which(self.cmd + ext)
            if cmd:
                break
        return cmd

    def __call__(self, mgr: LanguageServerManagerAPI) -> KeyedLanguageServerSpecs:
        cmd = self.solve()
        spec = dict(self.spec)
        if not cmd:
            troubleshooting = [f'{self.cmd} not found.']
            if 'troubleshoot' in spec:
                troubleshooting.append(spec['troubleshoot'])
            spec['troubleshoot'] = '\n\n'.join(troubleshooting)
        if not cmd and BUILDING_DOCS:
            cmd = self.cmd
        return {self.key: {'argv': [cmd, *self.args] if cmd else [self.cmd, *self.args], 'languages': self.languages, 'version': SPEC_VERSION, **spec}}