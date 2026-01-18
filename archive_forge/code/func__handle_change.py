import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def _handle_change():
    change_counter[0] += 1