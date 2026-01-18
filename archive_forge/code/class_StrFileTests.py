from io import StringIO
from twisted.python import text
from twisted.trial import unittest
class StrFileTests(unittest.TestCase):

    def setUp(self) -> None:
        self.io = StringIO('this is a test string')

    def tearDown(self) -> None:
        pass

    def test_1_f(self) -> None:
        self.assertFalse(text.strFile('x', self.io))

    def test_1_1(self) -> None:
        self.assertTrue(text.strFile('t', self.io))

    def test_1_2(self) -> None:
        self.assertTrue(text.strFile('h', self.io))

    def test_1_3(self) -> None:
        self.assertTrue(text.strFile('i', self.io))

    def test_1_4(self) -> None:
        self.assertTrue(text.strFile('s', self.io))

    def test_1_5(self) -> None:
        self.assertTrue(text.strFile('n', self.io))

    def test_1_6(self) -> None:
        self.assertTrue(text.strFile('g', self.io))

    def test_3_1(self) -> None:
        self.assertTrue(text.strFile('thi', self.io))

    def test_3_2(self) -> None:
        self.assertTrue(text.strFile('his', self.io))

    def test_3_3(self) -> None:
        self.assertTrue(text.strFile('is ', self.io))

    def test_3_4(self) -> None:
        self.assertTrue(text.strFile('ing', self.io))

    def test_3_f(self) -> None:
        self.assertFalse(text.strFile('bla', self.io))

    def test_large_1(self) -> None:
        self.assertTrue(text.strFile('this is a test', self.io))

    def test_large_2(self) -> None:
        self.assertTrue(text.strFile('is a test string', self.io))

    def test_large_f(self) -> None:
        self.assertFalse(text.strFile('ds jhfsa k fdas', self.io))

    def test_overlarge_f(self) -> None:
        self.assertFalse(text.strFile('djhsakj dhsa fkhsa s,mdbnfsauiw bndasdf hreew', self.io))

    def test_self(self) -> None:
        self.assertTrue(text.strFile('this is a test string', self.io))

    def test_insensitive(self) -> None:
        self.assertTrue(text.strFile('ThIs is A test STRING', self.io, False))