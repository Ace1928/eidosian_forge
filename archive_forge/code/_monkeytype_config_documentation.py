import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
Return a JitCallTraceStoreLogger that logs to the configured trace store.