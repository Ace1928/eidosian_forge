from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
class TestContentFilterContext(TestCase):

    def test_empty_filter_context(self):
        ctx = ContentFilterContext()
        self.assertEqual(None, ctx.relpath())

    def test_filter_context_with_path(self):
        ctx = ContentFilterContext('foo/bar')
        self.assertEqual('foo/bar', ctx.relpath())