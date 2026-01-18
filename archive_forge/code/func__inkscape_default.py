import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor
@default('inkscape')
def _inkscape_default(self):
    inkscape_path = which('inkscape')
    if inkscape_path is not None:
        return inkscape_path
    if sys.platform == 'darwin':
        if os.path.isfile(INKSCAPE_APP_v1):
            return INKSCAPE_APP_v1
        if os.path.isfile(INKSCAPE_APP):
            return INKSCAPE_APP
    if sys.platform == 'win32':
        wr_handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        try:
            rkey = winreg.OpenKey(wr_handle, 'SOFTWARE\\Classes\\inkscape.svg\\DefaultIcon')
            inkscape = winreg.QueryValueEx(rkey, '')[0]
        except FileNotFoundError:
            msg = 'Inkscape executable not found'
            raise FileNotFoundError(msg) from None
        return inkscape
    return 'inkscape'