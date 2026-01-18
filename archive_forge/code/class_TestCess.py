import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
class TestCess(unittest.TestCase):

    def test_catalan(self):
        words = cess_cat.words()[:15]
        txt = "El Tribunal_Suprem -Fpa- TS -Fpt- ha confirmat la condemna a quatre anys d' inhabilitació especial"
        self.assertEqual(words, txt.split())
        self.assertEqual(cess_cat.tagged_sents()[0][34][0], 'càrrecs')

    def test_esp(self):
        words = cess_esp.words()[:15]
        txt = 'El grupo estatal Electricité_de_France -Fpa- EDF -Fpt- anunció hoy , jueves , la compra del'
        self.assertEqual(words, txt.split())
        self.assertEqual(cess_esp.words()[115], 'años')