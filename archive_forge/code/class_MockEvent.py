import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
class MockEvent(object):

    def __init__(self, line):
        self.line = line