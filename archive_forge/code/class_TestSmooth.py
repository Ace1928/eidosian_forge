from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestSmooth(SoupTest):
    """Test Tag.smooth."""

    def test_smooth(self):
        soup = self.soup('<div>a</div>')
        div = soup.div
        div.append('b')
        div.append('c')
        div.append(Comment('Comment 1'))
        div.append(Comment('Comment 2'))
        div.append('d')
        builder = self.default_builder()
        span = Tag(soup, builder, 'span')
        span.append('1')
        span.append('2')
        div.append(span)
        assert None == div.span.string
        assert 7 == len(div.contents)
        div.smooth()
        assert 5 == len(div.contents)
        assert 'abc' == div.contents[0]
        assert '12' == div.span.string
        assert 'Comment 1' == div.contents[1]
        assert 'Comment 2' == div.contents[2]