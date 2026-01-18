import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetStripPostbuilds(self, configname, output_binary, quiet):
    """Returns a list of shell commands that contain the shell commands
    necessary to strip this target's binary. These should be run as postbuilds
    before the actual postbuilds run."""
    self.configname = configname
    result = []
    if self._Test('DEPLOYMENT_POSTPROCESSING', 'YES', default='NO') and self._Test('STRIP_INSTALLED_PRODUCT', 'YES', default='NO'):
        default_strip_style = 'debugging'
        if (self.spec['type'] == 'loadable_module' or self._IsIosAppExtension()) and self._IsBundle():
            default_strip_style = 'non-global'
        elif self.spec['type'] == 'executable':
            default_strip_style = 'all'
        strip_style = self._Settings().get('STRIP_STYLE', default_strip_style)
        strip_flags = {'all': '', 'non-global': '-x', 'debugging': '-S'}[strip_style]
        explicit_strip_flags = self._Settings().get('STRIPFLAGS', '')
        if explicit_strip_flags:
            strip_flags += ' ' + _NormalizeEnvVarReferences(explicit_strip_flags)
        if not quiet:
            result.append('echo STRIP\\(%s\\)' % self.spec['target_name'])
        result.append(f'strip {strip_flags} {output_binary}')
    self.configname = None
    return result