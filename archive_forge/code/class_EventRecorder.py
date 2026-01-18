from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
class EventRecorder(object):

    def stop(self) -> None:
        pass