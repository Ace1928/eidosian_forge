import platform
import time
import unittest
import pytest
from monty.functools import (
class TestProfMain:

    def test_prof_decorator(self):
        """Testing prof_main decorator."""
        import sys

        @prof_main
        def main():
            return sys.exit(1)
        _ = sys.argv[:]
        if len(sys.argv) == 1:
            sys.argv.append('prof')
        else:
            sys.argv[1] = 'prof'