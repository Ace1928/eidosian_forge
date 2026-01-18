from __future__ import (absolute_import, division, print_function)
import difflib
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.common.text.converters import to_text
def _print_task_result(self, result, error=False, **kwargs):
    """Run when a task finishes correctly."""
    if 'print_action' in result._task.tags or error or self._display.verbosity > 1:
        self._print_task()
        self.last_skipped = False
        msg = to_text(result._result.get('msg', '')) or to_text(result._result.get('reason', ''))
        stderr = [result._result.get('exception', None), result._result.get('module_stderr', None)]
        stderr = '\n'.join([e for e in stderr if e]).strip()
        self._print_host_or_item(result._host, result._result.get('changed', False), msg, result._result.get('diff', None), is_host=True, error=error, stdout=result._result.get('module_stdout', None), stderr=stderr.strip())
        if 'results' in result._result:
            for r in result._result['results']:
                failed = 'failed' in r and r['failed']
                stderr = [r.get('exception', None), r.get('module_stderr', None)]
                stderr = '\n'.join([e for e in stderr if e]).strip()
                self._print_host_or_item(r['item'], r.get('changed', False), to_text(r.get('msg', '')), r.get('diff', None), is_host=False, error=failed, stdout=r.get('module_stdout', None), stderr=stderr.strip())
    else:
        self.last_skipped = True
        print('.', end='')