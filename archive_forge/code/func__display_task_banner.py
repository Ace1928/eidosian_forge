from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
from ansible.utils.color import colorize, hostcolor
from ansible.utils.display import Display
import sys
def _display_task_banner(self):
    if not self.shown_title:
        self.shown_title = True
        sys.stdout.write(vt100.restore + vt100.reset + vt100.clearline + vt100.underline)
        sys.stdout.write('%s %d: %s' % (self.type, self.count[self.type], self.task.get_name().strip()))
        sys.stdout.write(vt100.restore + vt100.reset + '\n' + vt100.save + vt100.clearline)
        sys.stdout.flush()
    else:
        sys.stdout.write(vt100.restore + vt100.reset + vt100.clearline)
    self.keep = False