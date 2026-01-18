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
def _get_shebang(self, encoding, post_interp=b'', options=None):
    enquote = True
    if self.executable:
        executable = self.executable
        enquote = False
    elif not sysconfig.is_python_build():
        executable = get_executable()
    elif in_venv():
        executable = os.path.join(sysconfig.get_path('scripts'), 'python%s' % sysconfig.get_config_var('EXE'))
    elif os.name == 'nt':
        executable = os.path.join(sysconfig.get_config_var('BINDIR'), 'python%s' % sysconfig.get_config_var('EXE'))
    else:
        executable = os.path.join(sysconfig.get_config_var('BINDIR'), 'python%s%s' % (sysconfig.get_config_var('VERSION'), sysconfig.get_config_var('EXE')))
    if options:
        executable = self._get_alternate_executable(executable, options)
    if sys.platform.startswith('java'):
        executable = self._fix_jython_executable(executable)
    if enquote:
        executable = enquote_executable(executable)
    executable = executable.encode('utf-8')
    if sys.platform == 'cli' and '-X:Frames' not in post_interp and ('-X:FullFrames' not in post_interp):
        post_interp += b' -X:Frames'
    shebang = self._build_shebang(executable, post_interp)
    try:
        shebang.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('The shebang (%r) is not decodable from utf-8' % shebang)
    if encoding != 'utf-8':
        try:
            shebang.decode(encoding)
        except UnicodeDecodeError:
            raise ValueError('The shebang (%r) is not decodable from the script encoding (%r)' % (shebang, encoding))
    return shebang