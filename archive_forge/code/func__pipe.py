import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def _pipe(self):
    if pty:
        r, w = pty.openpty()
    else:
        r, w = os.pipe()
    return (r, w)