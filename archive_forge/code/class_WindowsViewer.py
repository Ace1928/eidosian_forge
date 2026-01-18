from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class WindowsViewer(Viewer):
    """The default viewer on Windows is the default system application for PNG files."""
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        return f'start "Pillow" /WAIT "{file}" && ping -n 4 127.0.0.1 >NUL && del /f "{file}"'