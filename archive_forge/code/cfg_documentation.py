import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
Returns the CFG accumulated so far and resets the builder.

    Returns:
      Graph
    