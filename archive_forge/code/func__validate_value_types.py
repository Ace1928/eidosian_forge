from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _validate_value_types(self, *, section='', option='', value=''):
    """Raises a TypeError for non-string values.

        The only legal non-string value if we allow valueless
        options is None, so we need to check if the value is a
        string if:
        - we do not allow valueless options, or
        - we allow valueless options but the value is not None

        For compatibility reasons this method is not used in classic set()
        for RawConfigParsers. It is invoked in every case for mapping protocol
        access and in ConfigParser.set().
        """
    if not isinstance(section, str):
        raise TypeError('section names must be strings')
    if not isinstance(option, str):
        raise TypeError('option keys must be strings')
    if not self._allow_no_value or value:
        if not isinstance(value, str):
            raise TypeError('option values must be strings')