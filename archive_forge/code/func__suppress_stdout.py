from __future__ import (absolute_import, division, print_function)
import sys
from contextlib import contextmanager
from ansible.template import Templar
from ansible.vars.manager import VariableManager
from ansible.plugins.callback.default import CallbackModule as Default
from ansible.module_utils.common.text.converters import to_text
@contextmanager
def _suppress_stdout(self, enabled):
    saved_stdout = sys.stdout
    if enabled:
        sys.stdout = DummyStdout()
    yield
    sys.stdout = saved_stdout