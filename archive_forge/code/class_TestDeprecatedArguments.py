from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestDeprecatedArguments(SoupTest):

    @pytest.mark.parametrize('method_name', ['find', 'find_all', 'find_parent', 'find_parents', 'find_next', 'find_all_next', 'find_previous', 'find_all_previous', 'find_next_sibling', 'find_next_siblings', 'find_previous_sibling', 'find_previous_siblings'])
    def test_find_type_method_string(self, method_name):
        soup = self.soup('<a>some</a><b>markup</b>')
        method = getattr(soup.b, method_name)
        with warnings.catch_warnings(record=True) as w:
            method(text='markup')
            [warning] = w
            assert warning.filename == __file__
            msg = str(warning.message)
            assert msg == "The 'text' argument to find()-type methods is deprecated. Use 'string' instead."

    def test_soupstrainer_constructor_string(self):
        with warnings.catch_warnings(record=True) as w:
            strainer = SoupStrainer(text='text')
            assert strainer.text == 'text'
            [warning] = w
            msg = str(warning.message)
            assert warning.filename == __file__
            assert msg == "The 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead."