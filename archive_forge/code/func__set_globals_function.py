import sys
import os
from _pydev_bundle._pydev_execfile import execfile
def _set_globals_function(get_globals):
    global _get_globals_callback
    _get_globals_callback = get_globals