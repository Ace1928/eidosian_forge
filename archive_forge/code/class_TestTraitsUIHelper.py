import unittest
from unittest import mock
from traits.testing.optional_dependencies import requires_traitsui
from traits.util._traitsui_helpers import check_traitsui_major_version
@requires_traitsui
class TestTraitsUIHelper(unittest.TestCase):

    def test_check_version_error(self):
        with mock.patch('traitsui.__version__', '6.1.2'):
            with self.assertRaises(RuntimeError) as exception_context:
                check_traitsui_major_version(7)
        self.assertEqual(str(exception_context.exception), "TraitsUI 7 or higher is required. Got version '6.1.2'")

    def test_check_version_okay(self):
        with mock.patch('traitsui.__version__', '7.0.0'):
            try:
                check_traitsui_major_version(7)
            except Exception:
                self.fail('Given TraitsUI version is okay, sanity check unexpectedly failed.')