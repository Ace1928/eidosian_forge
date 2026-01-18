from the command line:
import functools
import re
import types
import unittest
import uuid
def _IsSingletonList(testcases):
    """True iff testcases contains only a single non-tuple element."""
    return len(testcases) == 1 and (not isinstance(testcases[0], tuple))