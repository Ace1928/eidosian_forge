from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class GmDisplayViewer(UnixViewer):
    """The GraphicsMagick ``gm display`` command."""

    def get_command_ex(self, file, **options):
        executable = 'gm'
        command = 'gm display'
        return (command, executable)

    def show_file(self, path, **options):
        """
        Display given file.
        """
        subprocess.Popen(['gm', 'display', path])
        return 1