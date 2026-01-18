from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
class NullCrypter(Crypter):
    """A crypter that does nothing"""

    def encrypt(self, value):
        return value

    def decrypt(self, value):
        return value