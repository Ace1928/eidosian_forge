import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
class TestUdhr(unittest.TestCase):

    def test_words(self):
        for name in udhr.fileids():
            words = list(udhr.words(name))
            self.assertTrue(words)

    def test_raw_unicode(self):
        for name in udhr.fileids():
            txt = udhr.raw(name)
            assert not isinstance(txt, bytes), name

    def test_polish_encoding(self):
        text_pl = udhr.raw('Polish-Latin2')[:164]
        text_ppl = udhr.raw('Polish_Polski-Latin2')[:164]
        expected = 'POWSZECHNA DEKLARACJA PRAW CZŁOWIEKA\n[Preamble]\nTrzecia Sesja Ogólnego Zgromadzenia ONZ, obradująca w Paryżu, uchwaliła 10 grudnia 1948 roku jednomyślnie Powszechną'
        assert text_pl == expected, 'Polish-Latin2'
        assert text_ppl == expected, 'Polish_Polski-Latin2'