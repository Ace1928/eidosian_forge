import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor
@default('from_format')
def _from_format_default(self):
    return 'image/svg+xml'