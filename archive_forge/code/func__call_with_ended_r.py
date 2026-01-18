import gc
import multiprocessing
import os
import pickle
import pytest
from rpy2 import rinterface
import rpy2
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import signal
import sys
import subprocess
import tempfile
import textwrap
import time
def _call_with_ended_r(queue):
    import rpy2.rinterface_cffi as rinterface
    rinterface.initr()
    rdate = rinterface.baseenv['date']
    rinterface.endr(0)
    try:
        rdate()
        res = (False, None)
    except RuntimeError as re:
        res = (True, re)
    except Exception as e:
        res = (False, e)
    queue.put(res)