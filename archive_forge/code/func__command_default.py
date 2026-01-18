import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor
@default('command')
def _command_default(self):
    major_version = self.inkscape_version.split('.')[0]
    command = [self.inkscape]
    if int(major_version) < 1:
        command.append('--without-gui')
        command.append('--export-pdf={to_filename}')
    else:
        command.append('--export-filename={to_filename}')
    command.append('{from_filename}')
    return command