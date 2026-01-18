import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
class TestExceptionGlobster(TestCase):

    def test_exclusion_patterns(self):
        """test that exception patterns are not matched"""
        patterns = ['*', '!./local', '!./local/**/*', '!RE:\\.z.*', '!!./.zcompdump']
        globster = ExceptionGlobster(patterns)
        self.assertEqual('*', globster.match('tmp/foo.txt'))
        self.assertEqual(None, globster.match('local'))
        self.assertEqual(None, globster.match('local/bin/wombat'))
        self.assertEqual(None, globster.match('.zshrc'))
        self.assertEqual(None, globster.match('.zfunctions/fiddle/flam'))
        self.assertEqual('!!./.zcompdump', globster.match('.zcompdump'))

    def test_exclusion_order(self):
        """test that ordering of exclusion patterns does not matter"""
        patterns = ['static/**/*.html', '!static/**/versionable.html']
        globster = ExceptionGlobster(patterns)
        self.assertEqual('static/**/*.html', globster.match('static/foo.html'))
        self.assertEqual(None, globster.match('static/versionable.html'))
        self.assertEqual(None, globster.match('static/bar/versionable.html'))
        globster = ExceptionGlobster(reversed(patterns))
        self.assertEqual('static/**/*.html', globster.match('static/foo.html'))
        self.assertEqual(None, globster.match('static/versionable.html'))
        self.assertEqual(None, globster.match('static/bar/versionable.html'))