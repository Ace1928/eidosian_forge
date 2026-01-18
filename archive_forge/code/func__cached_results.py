import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
@memoize_method
def _cached_results(self, name, *args, **kwargs):
    return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)