import io
from .. import errors, i18n, tests, workingtree
def _check_exact(self, expected, source):
    self.assertEqual(expected, source)
    self.assertEqual(type(expected), type(source))