from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
class DummyCudaEventRecorder(EventRecorder):
    pass