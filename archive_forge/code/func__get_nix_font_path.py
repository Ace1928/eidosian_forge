import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_nix_font_path(self, name, style):
    proc = subprocess.Popen(['fc-list', '%s:style=%s' % (name, style), 'file'], stdout=subprocess.PIPE, stderr=None)
    stdout, _ = proc.communicate()
    if proc.returncode == 0:
        lines = stdout.splitlines()
        for line in lines:
            if line.startswith(b'Fontconfig warning:'):
                continue
            path = line.decode().strip().strip(':')
            if path:
                return path
        return None