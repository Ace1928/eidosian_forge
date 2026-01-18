import os
import re
import sys
from collections import namedtuple
from . import local
def _patch_eventlet():
    import eventlet.debug
    eventlet.monkey_patch()
    blockdetect = float(os.environ.get('EVENTLET_NOBLOCK', 0))
    if blockdetect:
        eventlet.debug.hub_blocking_detection(blockdetect, blockdetect)