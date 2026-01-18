import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class TestDiffFromToolEncodedFilename(tests.TestCaseWithTransport):

    def test_encodable_filename(self):
        diffobj = diff.DiffFromTool(['dummy', '{old_path}', '{new_path}'], None, None, None)
        for _, scenario in EncodingAdapter.encoding_scenarios:
            encoding = scenario['encoding']
            dirname = scenario['info']['directory']
            filename = scenario['info']['filename']
            self.overrideAttr(diffobj, '_fenc', lambda: encoding)
            relpath = dirname + '/' + filename
            fullpath = diffobj._safe_filename('safe', relpath)
            self.assertEqual(fullpath, fullpath.encode(encoding).decode(encoding))
            self.assertTrue(fullpath.startswith(diffobj._root + '/safe'))

    def test_unencodable_filename(self):
        diffobj = diff.DiffFromTool(['dummy', '{old_path}', '{new_path}'], None, None, None)
        for _, scenario in EncodingAdapter.encoding_scenarios:
            encoding = scenario['encoding']
            dirname = scenario['info']['directory']
            filename = scenario['info']['filename']
            if encoding == 'iso-8859-1':
                encoding = 'iso-8859-2'
            else:
                encoding = 'iso-8859-1'
            self.overrideAttr(diffobj, '_fenc', lambda: encoding)
            relpath = dirname + '/' + filename
            fullpath = diffobj._safe_filename('safe', relpath)
            self.assertEqual(fullpath, fullpath.encode(encoding).decode(encoding))
            self.assertTrue(fullpath.startswith(diffobj._root + '/safe'))