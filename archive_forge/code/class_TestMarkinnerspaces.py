import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
class TestMarkinnerspaces:

    def test_do_not_touch_normal_spaces(self):
        test_list = ['a ', ' a', 'a b c', "'abcdefghij'"]
        for i in test_list:
            assert markinnerspaces(i) == i

    def test_one_relevant_space(self):
        assert markinnerspaces("a 'b c' \\' \\'") == "a 'b@_@c' \\' \\'"
        assert markinnerspaces('a "b c" \\" \\"') == 'a "b@_@c" \\" \\"'

    def test_ignore_inner_quotes(self):
        assert markinnerspaces('a \'b c" " d\' e') == 'a \'b@_@c"@_@"@_@d\' e'
        assert markinnerspaces('a "b c\' \' d" e') == 'a "b@_@c\'@_@\'@_@d" e'

    def test_multiple_relevant_spaces(self):
        assert markinnerspaces("a 'b c' 'd e'") == "a 'b@_@c' 'd@_@e'"
        assert markinnerspaces('a "b c" "d e"') == 'a "b@_@c" "d@_@e"'