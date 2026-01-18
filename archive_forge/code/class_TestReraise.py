import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
class TestReraise(testtools.TestCase):
    """Tests for trivial reraise wrapper needed for Python 2/3 changes"""

    def test_exc_info(self):
        """After reraise exc_info matches plus some extra traceback"""
        try:
            raise ValueError('Bad value')
        except ValueError:
            _exc_info = sys.exc_info()
        try:
            reraise(*_exc_info)
        except ValueError:
            _new_exc_info = sys.exc_info()
        self.assertIs(_exc_info[0], _new_exc_info[0])
        self.assertIs(_exc_info[1], _new_exc_info[1])
        expected_tb = traceback.extract_tb(_exc_info[2])
        self.assertEqual(expected_tb, traceback.extract_tb(_new_exc_info[2])[-len(expected_tb):])

    def test_custom_exception_no_args(self):
        """Reraising does not require args attribute to contain params"""

        class CustomException(Exception):
            """Exception that expects and sets attrs but not args"""

            def __init__(self, value):
                Exception.__init__(self)
                self.value = value
        try:
            raise CustomException('Some value')
        except CustomException:
            _exc_info = sys.exc_info()
        self.assertRaises(CustomException, reraise, *_exc_info)