import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _restore_standard_output(self):
    """Restore the stdout and stderr streams."""
    sys.stdout, sys.stderr = (self.old_stdout, self.old_stderr)
    self.output_patched = False