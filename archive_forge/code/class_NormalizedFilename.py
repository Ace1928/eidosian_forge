import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
class NormalizedFilename(TestCaseWithTransport):
    """Test normalized_filename and associated helpers"""

    def test__accessible_normalized_filename(self):
        anf = osutils._accessible_normalized_filename
        self.assertEqual(('ascii', True), anf('ascii'))
        self.assertEqual((a_circle_c, True), anf(a_circle_c))
        self.assertEqual((a_circle_c, True), anf(a_circle_d))
        self.assertEqual((a_dots_c, True), anf(a_dots_c))
        self.assertEqual((a_dots_c, True), anf(a_dots_d))
        self.assertEqual((z_umlat_c, True), anf(z_umlat_c))
        self.assertEqual((z_umlat_c, True), anf(z_umlat_d))
        self.assertEqual((squared_c, True), anf(squared_c))
        self.assertEqual((squared_c, True), anf(squared_d))
        self.assertEqual((quarter_c, True), anf(quarter_c))
        self.assertEqual((quarter_c, True), anf(quarter_d))

    def test__inaccessible_normalized_filename(self):
        inf = osutils._inaccessible_normalized_filename
        self.assertEqual(('ascii', True), inf('ascii'))
        self.assertEqual((a_circle_c, True), inf(a_circle_c))
        self.assertEqual((a_circle_c, False), inf(a_circle_d))
        self.assertEqual((a_dots_c, True), inf(a_dots_c))
        self.assertEqual((a_dots_c, False), inf(a_dots_d))
        self.assertEqual((z_umlat_c, True), inf(z_umlat_c))
        self.assertEqual((z_umlat_c, False), inf(z_umlat_d))
        self.assertEqual((squared_c, True), inf(squared_c))
        self.assertEqual((squared_c, True), inf(squared_d))
        self.assertEqual((quarter_c, True), inf(quarter_c))
        self.assertEqual((quarter_c, True), inf(quarter_d))

    def test_functions(self):
        if osutils.normalizes_filenames():
            self.assertEqual(osutils.normalized_filename, osutils._accessible_normalized_filename)
        else:
            self.assertEqual(osutils.normalized_filename, osutils._inaccessible_normalized_filename)

    def test_platform(self):
        files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3']
        try:
            self.build_tree(files)
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        if sys.platform == 'darwin':
            expected = sorted([a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3'])
        else:
            expected = sorted(files)
        present = sorted(os.listdir('.'))
        self.assertEqual(expected, present)

    def test_access_normalized(self):
        files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3', squared_c + '.4', quarter_c + '.5']
        try:
            self.build_tree(files, line_endings='native')
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        for fname in files:
            path, can_access = osutils.normalized_filename(fname)
            self.assertEqual(path, fname)
            self.assertTrue(can_access)
            with open(path, 'rb') as f:
                shouldbe = b'contents of %s%s' % (path.encode('utf8'), os.linesep.encode('utf-8'))
                actual = f.read()
            self.assertEqual(shouldbe, actual, 'contents of %r is incorrect: %r != %r' % (path, shouldbe, actual))

    def test_access_non_normalized(self):
        files = [a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3']
        try:
            self.build_tree(files)
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        for fname in files:
            path, can_access = osutils.normalized_filename(fname)
            self.assertNotEqual(path, fname)
            f = open(fname, 'rb')
            f.close()
            if can_access:
                f = open(path, 'rb')
                f.close()
            else:
                self.assertRaises(IOError, open, path, 'rb')