from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def _addjunk(chunks):
    return [b'junk\n'] + [s for s in chunks]