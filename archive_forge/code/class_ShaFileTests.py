import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class ShaFileTests(TestCase):

    def test_deflated_smaller_window_buffer(self):
        sf = ShaFile.from_file(BytesIO(small_buffer_zlib_object))
        self.assertEqual(sf.type_name, b'tag')
        self.assertEqual(sf.tagger, b' <@localhost>')