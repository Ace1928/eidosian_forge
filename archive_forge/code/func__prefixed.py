import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def _prefixed(prefix: str, s: str) -> str:
    """Prefixes a string (if not empty)"""
    return prefix + s if len(s) > 0 else s