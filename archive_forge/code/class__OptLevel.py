import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
class _OptLevel(int):
    """This class holds the "optimisation level" set in `NUMBA_OPT`. As this env
    var can be an int or a string, but is almost always interpreted as an int,
    this class subclasses int so as to get the common behaviour but stores the
    actual value as a `_raw_value` member. The value "max" is a special case
    and the property `is_opt_max` can be queried to find if the optimisation
    level (supplied value at construction time) is "max"."""

    def __new__(cls, *args, **kwargs):
        assert len(args) == 1
        value, = args
        _int_value = 3 if value == 'max' else int(value)
        new = super().__new__(cls, _int_value, **kwargs)
        new._raw_value = value if value == 'max' else _int_value
        return new

    @property
    def is_opt_max(self):
        """Returns True if the the optimisation level is "max" False
        otherwise."""
        return self._raw_value == 'max'

    def __repr__(self):
        if isinstance(self._raw_value, str):
            arg = f"'{self._raw_value}'"
        else:
            arg = self._raw_value
        return f'_OptLevel({arg})'