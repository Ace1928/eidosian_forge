import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
class JitTypeTraceStore:

    def __init__(self):
        self.trace_records = None