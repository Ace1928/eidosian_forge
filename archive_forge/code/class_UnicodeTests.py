from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class UnicodeTests(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        ppu = pp.pyparsing_unicode
        ppc = pp.pyparsing_common
        kanji_printables = ppu.Japanese.Kanji.printables
        katakana_printables = ppu.Japanese.Katakana.printables
        hiragana_printables = ppu.Japanese.Hiragana.printables
        japanese_printables = ppu.Japanese.printables
        self.assertEqual(set(japanese_printables), set(kanji_printables + katakana_printables + hiragana_printables), 'failed to construct ranges by merging Japanese types')
        cjk_printables = ppu.CJK.printables
        self.assertEqual(len(cjk_printables), len(set(cjk_printables)), 'CJK contains duplicate characters - all should be unique')
        chinese_printables = ppu.Chinese.printables
        korean_printables = ppu.Korean.printables
        print_(len(cjk_printables), len(set(chinese_printables + korean_printables + japanese_printables)))
        self.assertEqual(len(cjk_printables), len(set(chinese_printables + korean_printables + japanese_printables)), 'failed to construct ranges by merging Chinese, Japanese and Korean')
        alphas = ppu.Greek.alphas
        greet = pp.Word(alphas) + ',' + pp.Word(alphas) + '!'
        hello = u'Καλημέρα, κόσμε!'
        result = greet.parseString(hello)
        print_(result)
        self.assertTrue(result.asList() == [u'Καλημέρα', ',', u'κόσμε', '!'], "Failed to parse Greek 'Hello, World!' using pyparsing_unicode.Greek.alphas")

        class Turkish_set(ppu.Latin1, ppu.LatinA):
            pass
        self.assertEqual(set(Turkish_set.printables), set(ppu.Latin1.printables + ppu.LatinA.printables), 'failed to construct ranges by merging Latin1 and LatinA (printables)')
        self.assertEqual(set(Turkish_set.alphas), set(ppu.Latin1.alphas + ppu.LatinA.alphas), 'failed to construct ranges by merging Latin1 and LatinA (alphas)')
        self.assertEqual(set(Turkish_set.nums), set(ppu.Latin1.nums + ppu.LatinA.nums), 'failed to construct ranges by merging Latin1 and LatinA (nums)')
        key = pp.Word(Turkish_set.alphas)
        value = ppc.integer | pp.Word(Turkish_set.alphas, Turkish_set.alphanums)
        EQ = pp.Suppress('=')
        key_value = key + EQ + value
        sample = u'            şehir=İzmir\n            ülke=Türkiye\n            nüfus=4279677'
        result = pp.Dict(pp.OneOrMore(pp.Group(key_value))).parseString(sample)
        print_(result.asDict())
        self.assertEqual(result.asDict(), {u'şehir': u'İzmir', u'ülke': u'Türkiye', u'nüfus': 4279677}, 'Failed to parse Turkish key-value pairs')