from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
from ansible.utils.color import colorize, hostcolor
from ansible.utils.display import Display
import sys
def _display_results(self, result, status):
    if self._display.verbosity == 0 and self.keep:
        sys.stdout.write(vt100.restore + vt100.reset + '\n' + vt100.save + vt100.clearline)
    else:
        sys.stdout.write(vt100.restore + vt100.reset + vt100.clearline)
    self.keep = False
    self._clean_results(result._result)
    dump = ''
    if result._task.action == 'include':
        return
    elif status == 'ok':
        return
    elif status == 'ignored':
        dump = self._handle_exceptions(result._result)
    elif status == 'failed':
        dump = self._handle_exceptions(result._result)
    elif status == 'unreachable':
        dump = result._result['msg']
    if not dump:
        dump = self._dump_results(result._result)
    if result._task.loop and 'results' in result._result:
        self._process_items(result)
    else:
        sys.stdout.write(colors[status] + status + ': ')
        delegated_vars = result._result.get('_ansible_delegated_vars', None)
        if delegated_vars:
            sys.stdout.write(vt100.reset + result._host.get_name() + '>' + colors[status] + delegated_vars['ansible_host'])
        else:
            sys.stdout.write(result._host.get_name())
        sys.stdout.write(': ' + dump + '\n')
        sys.stdout.write(vt100.reset + vt100.save + vt100.clearline)
        sys.stdout.flush()
    if status == 'changed':
        self._handle_warnings(result._result)