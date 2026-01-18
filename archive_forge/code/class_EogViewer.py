from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class EogViewer(UnixViewer):
    """The GNOME Image Viewer ``eog`` command."""

    def get_command_ex(self, file, **options):
        executable = 'eog'
        command = 'eog -n'
        return (command, executable)

    def show_file(self, path, **options):
        """
        Display given file.
        """
        subprocess.Popen(['eog', '-n', path])
        return 1