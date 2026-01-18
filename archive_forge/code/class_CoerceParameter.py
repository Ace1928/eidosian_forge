from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CoerceParameter:
    """
    Utility class that can corce a parameter before storing it.
    """

    def __init__(self, options, coerce):
        """
        @param options: parent Options object
        @param coerce: callable used to coerce the value.
        """
        self.options = options
        self.coerce = coerce
        self.doc = getattr(self.coerce, 'coerceDoc', '')

    def dispatch(self, parameterName, value):
        """
        When called in dispatch, do the coerce for C{value} and save the
        returned value.
        """
        if value is None:
            raise UsageError(f"Parameter '{parameterName}' requires an argument.")
        try:
            value = self.coerce(value)
        except ValueError as e:
            raise UsageError(f'Parameter type enforcement failed: {e}')
        self.options.opts[parameterName] = value