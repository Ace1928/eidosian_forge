import functools
import sys
import types
import warnings
import unittest
class Trap:

    def __get__(*ignored):
        self.fail('Non-test attribute accessed')