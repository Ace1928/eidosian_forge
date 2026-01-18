import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
class TestCaseWithMemoryTransportAndScript(tests.TestCaseWithMemoryTransport):
    """Helper class to experiment shell-like test and memory fs.

    This not intended to be used outside of experiments in implementing memoy
    based file systems and evolving bzr so that test can use only memory based
    resources.
    """

    def setUp(self):
        super().setUp()
        self.script_runner = ScriptRunner()
        self.overrideEnv('INSIDE_EMACS', '1')

    def run_script(self, script, null_output_matches_anything=False):
        return self.script_runner.run_script(self, script, null_output_matches_anything=null_output_matches_anything)

    def run_command(self, cmd, input, output, error):
        return self.script_runner.run_command(self, cmd, input, output, error)