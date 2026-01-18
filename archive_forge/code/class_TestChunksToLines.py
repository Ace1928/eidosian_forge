import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestChunksToLines(tests.TestCase):

    def test_smoketest(self):
        self.assertEqual([b'foo\n', b'bar\n', b'baz\n'], osutils.chunks_to_lines([b'foo\nbar', b'\nbaz\n']))
        self.assertEqual([b'foo\n', b'bar\n', b'baz\n'], osutils.chunks_to_lines([b'foo\n', b'bar\n', b'baz\n']))

    def test_osutils_binding(self):
        from . import test__chunks_to_lines
        if test__chunks_to_lines.compiled_chunkstolines_feature.available():
            from .._chunks_to_lines_pyx import chunks_to_lines
        else:
            from .._chunks_to_lines_py import chunks_to_lines
        self.assertIs(chunks_to_lines, osutils.chunks_to_lines)