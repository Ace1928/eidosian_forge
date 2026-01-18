import concurrent.futures._base
import logging
import reprlib
import sys
import traceback
from . import events
def _check_cancel_other(f):
    if f.cancelled():
        fut.cancel()