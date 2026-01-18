from __future__ import (absolute_import, division, print_function)
import difflib
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.common.text.converters import to_text
def _print_task(self, task_name=None):
    if task_name is None:
        task_name = self.last_task_name
    if not self.printed_last_task:
        self.printed_last_task = True
        line_length = 120
        if self.last_skipped:
            print()
        line = '# {0} '.format(task_name)
        msg = colorize('{0}{1}'.format(line, '*' * (line_length - len(line))), 'bold')
        print(msg)