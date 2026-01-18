import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
class TestPatienceDiffLibFiles_c(TestPatienceDiffLibFiles):

    def setUp(self):
        super().setUp()
        try:
            from . import _patiencediff_c
        except ImportError:
            self.skipTest('C extension not built')
        self._PatienceSequenceMatcher = _patiencediff_c.PatienceSequenceMatcher_c