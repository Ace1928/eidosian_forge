import torch
import random
import os
import queue
from dataclasses import dataclass
from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
class ManagerWatchdog:

    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead