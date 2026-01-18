import os
import shutil
import sys
from pathlib import Path
from subprocess import check_output
from typing import List, Text, Union
from ..schema import SPEC_VERSION
from ..types import (
class NodeModuleSpec(SpecBase):
    """Helper for a nodejs-based language server spec in one of several
    node_modules
    """
    node_module = ''
    script: List[Text] = []

    def is_installed(self, mgr: LanguageServerManagerAPI) -> bool:
        node_module = self.solve(mgr)
        return bool(node_module)

    def solve(self, mgr: LanguageServerManagerAPI):
        return mgr.find_node_module(self.node_module, *self.script)

    def __call__(self, mgr: LanguageServerManagerAPI) -> KeyedLanguageServerSpecs:
        node_module = self.solve(mgr)
        spec = dict(self.spec)
        troubleshooting = ['Node.js is required to install this server.']
        if 'troubleshoot' in spec:
            troubleshooting.append(spec['troubleshoot'])
        spec['troubleshoot'] = '\n\n'.join(troubleshooting)
        is_installed = self.is_installed(mgr)
        return {self.key: {'argv': [mgr.nodejs, node_module, *self.args] if is_installed else [], 'languages': self.languages, 'version': SPEC_VERSION, **spec}}