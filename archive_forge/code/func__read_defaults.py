from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _read_defaults(self, defaults):
    """Reads the defaults passed in the initializer, implicitly converting
        values to strings like the rest of the API.

        Does not perform interpolation for backwards compatibility.
        """
    try:
        hold_interpolation = self._interpolation
        self._interpolation = Interpolation()
        self.read_dict({self.default_section: defaults})
    finally:
        self._interpolation = hold_interpolation