import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ClassicalDiv(ArithmeticBinaryOp):
    """
    The DIV instruction.
    """
    op = 'DIV'