import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
class TestMain(IntegrationTests):
    """
    Tests of the pyflakes main function.
    """

    def runPyflakes(self, paths, stdin=None):
        try:
            with SysStreamCapturing(stdin) as capture:
                main(args=paths)
        except SystemExit as e:
            self.assertIsInstance(e.code, bool)
            rv = int(e.code)
            return (capture.output, capture.error, rv)
        else:
            raise RuntimeError('SystemExit not raised')