import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor
@default('inkscape_version')
def _inkscape_version_default(self):
    p = subprocess.Popen([self.inkscape, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = p.communicate()
    if p.returncode != 0:
        msg = 'Unable to find inkscape executable --version'
        raise RuntimeError(msg)
    return output.decode('utf-8').split(' ')[1]