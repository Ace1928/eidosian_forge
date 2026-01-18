from __future__ import (absolute_import, division, print_function)
from ansible.plugins.callback import CallbackBase
from ansible import constants as C
def _command_generic_msg(self, hostname, result, caption):
    stdout = result.get('stdout', '').replace('\n', '\\n').replace('\r', '\\r')
    if 'stderr' in result and result['stderr']:
        stderr = result.get('stderr', '').replace('\n', '\\n').replace('\r', '\\r')
        return '%s | %s | rc=%s | (stdout) %s (stderr) %s' % (hostname, caption, result.get('rc', -1), stdout, stderr)
    else:
        return '%s | %s | rc=%s | (stdout) %s' % (hostname, caption, result.get('rc', -1), stdout)