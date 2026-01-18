from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
class _DeprecatedValue:

    def __init__(self, value: object, message: str, warning_class):
        self.value = value
        self.message = message
        self.warning_class = warning_class