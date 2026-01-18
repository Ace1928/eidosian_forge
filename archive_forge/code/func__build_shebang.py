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
def _build_shebang(self, executable, post_interp):
    """
        Build a shebang line. In the simple case (on Windows, or a shebang line
        which is not too long or contains spaces) use a simple formulation for
        the shebang. Otherwise, use /bin/sh as the executable, with a contrived
        shebang which allows the script to run either under Python or sh, using
        suitable quoting. Thanks to Harald Nordgren for his input.

        See also: http://www.in-ulm.de/~mascheck/various/shebang/#length
                  https://hg.mozilla.org/mozilla-central/file/tip/mach
        """
    if os.name != 'posix':
        simple_shebang = True
    else:
        shebang_length = len(executable) + len(post_interp) + 3
        if sys.platform == 'darwin':
            max_shebang_length = 512
        else:
            max_shebang_length = 127
        simple_shebang = b' ' not in executable and shebang_length <= max_shebang_length
    if simple_shebang:
        result = b'#!' + executable + post_interp + b'\n'
    else:
        result = b'#!/bin/sh\n'
        result += b"'''exec' " + executable + post_interp + b' "$0" "$@"\n'
        result += b"' '''"
    return result