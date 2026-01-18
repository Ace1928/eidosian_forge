from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def _deljunk(chunks, context):
    return [s for s in chunks[1:]]