from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
class TestFilteredSha(TestCaseInTempDir):

    def test_filtered_size_sha(self):
        text = b'Foo Bar Baz\n'
        with open('a', 'wb') as a:
            a.write(text)
        post_filtered_content = b''.join(_swapcase([text], None))
        expected_len = len(post_filtered_content)
        expected_sha = sha_string(post_filtered_content)
        self.assertEqual((expected_len, expected_sha), internal_size_sha_file_byname('a', [ContentFilter(_swapcase, _swapcase)]))