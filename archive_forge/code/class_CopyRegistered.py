from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class CopyRegistered:
    """
    A class that is pickleable only because it is registered with the
    C{copyreg} module.
    """

    def __init__(self) -> None:
        """
        Ensure that this object is normally not pickleable.
        """
        self.notPickleable = NotPickleable()