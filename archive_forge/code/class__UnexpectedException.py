from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
class _UnexpectedException(Exception):
    """An exception used to test HyperlinkTestCase.assertRaises."""