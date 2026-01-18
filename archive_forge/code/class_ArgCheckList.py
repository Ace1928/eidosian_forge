import unittest
from traits.api import (
class ArgCheckList(ArgCheckBase):
    value = List(Int, [0, 1, 2])