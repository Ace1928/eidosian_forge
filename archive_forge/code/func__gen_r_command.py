import os
from shutil import which
from .. import config
from .base import (
def _gen_r_command(self, argstr, script_lines):
    """Generates commands and, if rfile specified, writes it to disk."""
    if not self.inputs.rfile:
        script = '; '.join([line for line in script_lines.split('\n') if not line.strip().startswith('#')])
        script = script.replace('"', '\\"')
        script = script.replace('$', '\\$')
    else:
        script_path = os.path.join(os.getcwd(), self.inputs.script_file)
        with open(script_path, 'wt') as rfile:
            rfile.write(script_lines)
        script = "source('%s')" % script_path
    return argstr % script