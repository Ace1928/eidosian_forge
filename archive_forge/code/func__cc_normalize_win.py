import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _cc_normalize_win(self, flags):
    for i, f in enumerate(reversed(flags)):
        if not re.match(self._cc_normalize_win_mrgx, f):
            continue
        i += 1
        return list(filter(self._cc_normalize_win_frgx.search, flags[:-i])) + flags[-i:]
    return flags