import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
def _actually(something):
    """
        A decorator that returns its argument rather than the thing it is
        decorating.

        This allows the documentation generator to see an alias for a method or
        constant as an object with a docstring and thereby document it and
        allow references to it statically.

        @param something: An object to create an alias for.
        @type something: L{object}

        @return: a 1-argument callable that returns C{something}
        @rtype: L{object}
        """

    def decorate(thingWithADocstring):
        return something
    return decorate