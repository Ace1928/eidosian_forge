from __future__ import (absolute_import, division, print_function)
import re
from ansible.plugins.terminal import TerminalBase
from ansible.errors import AnsibleConnectionFailure
class TerminalModule(TerminalBase):
    terminal_stdout_re = [re.compile(b'[\\r\\n]?(?:\\([^\\)]+\\)){,5}(?:>|#)\\s*$'), re.compile(b'[\\r\\n]?[\\w+\\-\\.:\\/\\[\\]]+(?:\\([^\\)]+\\)){,3}(?:>|#)\\s*$'), re.compile(b'\\[\\w+\\@[\\w\\-\\.]+(?: [^\\]])\\] ?[>#\\$]\\s*$'), re.compile(b'(?:new|confirm) password:')]
    terminal_stderr_re = [re.compile(b'% ?Error'), re.compile(b'Syntax Error', re.I), re.compile(b'% User not present'), re.compile(b'% ?Bad secret'), re.compile(b'invalid input', re.I), re.compile(b'(?:incomplete|ambiguous) command', re.I), re.compile(b'connection timed out', re.I), re.compile(b'the new password was not confirmed', re.I), re.compile(b'[^\\r\\n]+ not found', re.I), re.compile(b"'[^']' +returned error code: ?\\d+"), re.compile(b'[^\\r\\n]\\/bin\\/(?:ba)?sh')]

    def on_open_shell(self):
        try:
            self._exec_cli_command(b'modify cli preference display-threshold 0 pager disabled')
            self._exec_cli_command(b'run /util bash -c "stty cols 1000000" 2> /dev/null')
        except AnsibleConnectionFailure as ex:
            output = str(ex)
            if 'modify: command not found' in output:
                try:
                    self._exec_cli_command(b'tmsh modify cli preference display-threshold 0 pager disabled')
                    self._exec_cli_command(b'stty cols 1000000 2> /dev/null')
                except AnsibleConnectionFailure:
                    raise AnsibleConnectionFailure('unable to set terminal parameters')