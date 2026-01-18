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
def add_fd(self, fd):
    if not self._fds:
        self._register()
    self._fds.add(fd)
    self.handle_window_size_change()