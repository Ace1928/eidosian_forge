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
def _is_shell(self, executable):
    """
            Determine if the specified executable is a script
            (contains a #! line)
            """
    try:
        with open(executable) as fp:
            return fp.read(2) == '#!'
    except (OSError, IOError):
        logger.warning('Failed to open %s', executable)
        return False