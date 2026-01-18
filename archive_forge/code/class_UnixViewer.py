from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class UnixViewer(Viewer):
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        command = self.get_command_ex(file, **options)[0]
        return f'({command} {quote(file)}'