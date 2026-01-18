from io import BytesIO
import logging
import os
import re
import struct
import sys
import time
from zipfile import ZipInfo
from .compat import sysconfig, detect_encoding, ZipFile
from .resources import finder
from .util import (FileOperator, get_export_entry, convert_path,
import re
import sys
from %(module)s import %(import_name)s
def _fix_jython_executable(self, executable):
    if self._is_shell(executable):
        import java
        if java.lang.System.getProperty('os.name') == 'Linux':
            return executable
    elif executable.lower().endswith('jython.exe'):
        return executable
    return '/usr/bin/env %s' % executable