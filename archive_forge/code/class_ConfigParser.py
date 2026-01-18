from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class ConfigParser(RawConfigParser):
    """ConfigParser implementing interpolation."""
    _DEFAULT_INTERPOLATION = BasicInterpolation()

    def set(self, section, option, value=None):
        """Set an option.  Extends RawConfigParser.set by validating type and
        interpolation syntax on the value."""
        self._validate_value_types(option=option, value=value)
        super().set(section, option, value)

    def add_section(self, section):
        """Create a new section in the configuration.  Extends
        RawConfigParser.add_section by validating if the section name is
        a string."""
        self._validate_value_types(section=section)
        super().add_section(section)

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