import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
class TreeBuilderSmokeTest(object):

    @pytest.mark.parametrize('multi_valued_attributes', [None, {}, dict(b=['class']), {'*': ['notclass']}])
    def test_attribute_not_multi_valued(self, multi_valued_attributes):
        markup = '<html xmlns="http://www.w3.org/1999/xhtml"><a class="a b c"></html>'
        soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
        assert soup.a['class'] == 'a b c'

    @pytest.mark.parametrize('multi_valued_attributes', [dict(a=['class']), {'*': ['class']}])
    def test_attribute_multi_valued(self, multi_valued_attributes):
        markup = '<a class="a b c">'
        soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
        assert soup.a['class'] == ['a', 'b', 'c']

    def test_invalid_doctype(self):
        markup = '<![if word]>content<![endif]>'
        markup = '<!DOCTYPE html]ff>'
        soup = self.soup(markup)