import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
def _make_env():
    env = os.environ.copy()
    env['PATH'] = env.get('PATH', '') + ':' + os.path.join(paste_parent, 'scripts') + ':' + os.path.join(paste_parent, 'paste', '3rd-party', 'sqlobject-files', 'scripts')
    env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + paste_parent
    return env