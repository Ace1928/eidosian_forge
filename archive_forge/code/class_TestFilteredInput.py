from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
class TestFilteredInput(TestCase):

    def test_filtered_input_file(self):
        external = b''.join(_sample_external)
        f = BytesIO(external)
        fileobj, size = filtered_input_file(f, [])
        self.assertEqual((external, 12), (fileobj.read(), size))
        f = BytesIO(external)
        expected = b''.join(_internal_1)
        fileobj, size = filtered_input_file(f, _stack_1)
        self.assertEqual((expected, 12), (fileobj.read(), size))
        f = BytesIO(external)
        expected = b''.join(_internal_2)
        fileobj, size = filtered_input_file(f, _stack_2)
        self.assertEqual((expected, 17), (fileobj.read(), size))