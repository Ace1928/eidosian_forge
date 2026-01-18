from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _GetModes(self):
    """Syncs the user settable modes of interest to the Coshell.

    Calls self._set_modes_callback if it was specified and any mode changed.
    """
    changed = False
    if self.Run('set -o | grep -q "^vi.*on"', check_modes=False) == 0:
        if self._edit_mode != 'vi':
            changed = True
            self._edit_mode = 'vi'
    elif self._edit_mode != 'emacs':
        changed = True
        self._edit_mode = 'emacs'
    ignore_eof = self._ignore_eof
    self._ignore_eof = self.Run('set -o | grep -q "^ignoreeof.*on"', check_modes=False) == 0
    if self._ignore_eof != ignore_eof:
        changed = True
    if changed and self._set_modes_callback:
        self._set_modes_callback()