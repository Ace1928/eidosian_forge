import unittest
from os import sys, path
class TestConfFeatures(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        unittest.TestCase.__init__(self, methodName)
        self._setup()

    def _setup(self):
        FakeCCompilerOpt.conf_nocache = True

    def test_features(self):
        for arch, compilers in arch_compilers.items():
            for cc in compilers:
                FakeCCompilerOpt.fake_info = (arch, cc, '')
                _TestConfFeatures()