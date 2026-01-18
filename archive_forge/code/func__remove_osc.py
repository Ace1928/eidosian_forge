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
def _remove_osc(self, text):
    return re.sub(ANSI_OSC_RE, '', text)