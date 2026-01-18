import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentWord(LineTestCase):

    def setUp(self):
        self.func = current_word

    def test_simple(self):
        self.assertAccess('|')
        self.assertAccess('|asdf')
        self.assertAccess('<a|sdf>')
        self.assertAccess('<asdf|>')
        self.assertAccess('<asdfg|>')
        self.assertAccess('asdf + <asdfg|>')
        self.assertAccess('<asdfg|> + asdf')

    def test_inside(self):
        self.assertAccess('<asd|>')
        self.assertAccess('<asd|fg>')

    def test_dots(self):
        self.assertAccess('<Object.attr1|>')
        self.assertAccess('<Object.attr1.attr2|>')
        self.assertAccess('<Object.att|r1.attr2>')
        self.assertAccess('stuff[stuff] + {123: 456} + <Object.attr1.attr2|>')
        self.assertAccess('stuff[<asd|fg>]')
        self.assertAccess('stuff[asdf[<asd|fg>]')

    def test_non_dots(self):
        self.assertAccess('].asdf|')
        self.assertAccess(').asdf|')
        self.assertAccess('foo[0].asdf|')
        self.assertAccess('foo().asdf|')
        self.assertAccess('foo().|')
        self.assertAccess('foo().asdf.|')
        self.assertAccess('foo[0].asdf.|')

    def test_open_paren(self):
        self.assertAccess('<foo(|>')