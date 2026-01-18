import os
from breezy.tests import TestCaseWithTransport
def _check_ancestry(self, location='', result=None):
    out = self.run_bzr(['ancestry', location])[0]
    if result is not None:
        self.assertEqualDiff(result, out)
    else:
        result = 'A1\nB1\nA2\nA3\n'
        if result != out:
            result = 'A1\nA2\nB1\nA3\n'
        self.assertEqualDiff(result, out)