import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentExpressionAttribute(LineTestCase):

    def setUp(self):
        self.func = current_expression_attribute

    def test_simple(self):
        self.assertAccess('Object.<attr1|>.')
        self.assertAccess('Object.<|attr1>.')
        self.assertAccess('Object.(|)')
        self.assertAccess('Object.another.(|)')
        self.assertAccess('asdf asdf asdf.(abc|)')

    def test_without_dot(self):
        self.assertAccess('Object|')
        self.assertAccess('Object|.')
        self.assertAccess('|Object.')

    def test_with_whitespace(self):
        self.assertAccess('Object. <attr|>')
        self.assertAccess('Object .<attr|>')
        self.assertAccess('Object . <attr|>')
        self.assertAccess('Object .asdf attr|')
        self.assertAccess('Object .<asdf|> attr')
        self.assertAccess('Object. asdf attr|')
        self.assertAccess('Object. <asdf|> attr')
        self.assertAccess('Object . asdf attr|')
        self.assertAccess('Object . <asdf|> attr')

    def test_indexing(self):
        self.assertAccess('abc[def].<ghi|>')
        self.assertAccess('abc[def].<|ghi>')
        self.assertAccess('abc[def].<gh|i>')
        self.assertAccess('abc[def].gh |i')
        self.assertAccess('abc[def]|')

    def test_strings(self):
        self.assertAccess('"hey".<a|>')
        self.assertAccess('"hey"|')
        self.assertAccess('"hey"|.a')
        self.assertAccess('"hey".<a|b>')
        self.assertAccess('"hey".asdf d|')
        self.assertAccess('"hey".<|>')