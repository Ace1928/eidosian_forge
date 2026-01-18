import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def _test_platforms(self, input, testval):
    utils._sys_platform = input
    pf = _get_platform()
    self.assertTrue(pf == testval)