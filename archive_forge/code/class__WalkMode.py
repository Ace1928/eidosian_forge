import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
class _WalkMode(enum.Enum):
    FORWARD = 1
    REVERSE = 2