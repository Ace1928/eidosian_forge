from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class InterpolationDepthError(InterpolationError):
    """Raised when substitutions are nested too deeply."""

    def __init__(self, option, section, rawval):
        msg = 'Recursion limit exceeded in value substitution: option {!r} in section {!r} contains an interpolation key which cannot be substituted in {} steps. Raw value: {!r}'.format(option, section, MAX_INTERPOLATION_DEPTH, rawval)
        InterpolationError.__init__(self, option, section, msg)
        self.args = (option, section, rawval)