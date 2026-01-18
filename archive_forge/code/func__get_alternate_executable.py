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
def _get_alternate_executable(self, executable, options):
    if options.get('gui', False) and self._is_nt:
        dn, fn = os.path.split(executable)
        fn = fn.replace('python', 'pythonw')
        executable = os.path.join(dn, fn)
    return executable