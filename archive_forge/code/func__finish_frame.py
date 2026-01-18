from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def _finish_frame(self):
    """
        Execute until the function returns to a relevant caller.
        """
    while True:
        result = self._finish()
        try:
            frame = gdb.selected_frame()
        except RuntimeError:
            break
        hitbp = re.search('Breakpoint (\\d+)', result)
        is_relevant = self.lang_info.is_relevant_function(frame)
        if hitbp or is_relevant or self.stopped():
            break
    return result