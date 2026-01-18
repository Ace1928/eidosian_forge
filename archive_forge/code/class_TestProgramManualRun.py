import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
class TestProgramManualRun(unittest.TestProgram):
    """A TestProgram which runs the tests manually."""

    def runTests(self, do_run=False):
        """Run the tests."""
        if do_run:
            unittest.TestProgram.runTests(self)