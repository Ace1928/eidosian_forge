import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
class AutoEscapeTest(unittest.TestCase):

    def setUp(self):
        self.templates = {'escaped.html': '{% autoescape xhtml_escape %}{{ name }}', 'unescaped.html': '{% autoescape None %}{{ name }}', 'default.html': '{{ name }}', 'include.html': "escaped: {% include 'escaped.html' %}\nunescaped: {% include 'unescaped.html' %}\ndefault: {% include 'default.html' %}\n", 'escaped_block.html': '{% autoescape xhtml_escape %}{% block name %}base: {{ name }}{% end %}', 'unescaped_block.html': '{% autoescape None %}{% block name %}base: {{ name }}{% end %}', 'escaped_extends_unescaped.html': '{% autoescape xhtml_escape %}{% extends "unescaped_block.html" %}', 'escaped_overrides_unescaped.html': '{% autoescape xhtml_escape %}{% extends "unescaped_block.html" %}{% block name %}extended: {{ name }}{% end %}', 'unescaped_extends_escaped.html': '{% autoescape None %}{% extends "escaped_block.html" %}', 'unescaped_overrides_escaped.html': '{% autoescape None %}{% extends "escaped_block.html" %}{% block name %}extended: {{ name }}{% end %}', 'raw_expression.html': '{% autoescape xhtml_escape %}expr: {{ name }}\nraw: {% raw name %}'}

    def test_default_off(self):
        loader = DictLoader(self.templates, autoescape=None)
        name = 'Bobby <table>s'
        self.assertEqual(loader.load('escaped.html').generate(name=name), b'Bobby &lt;table&gt;s')
        self.assertEqual(loader.load('unescaped.html').generate(name=name), b'Bobby <table>s')
        self.assertEqual(loader.load('default.html').generate(name=name), b'Bobby <table>s')
        self.assertEqual(loader.load('include.html').generate(name=name), b'escaped: Bobby &lt;table&gt;s\nunescaped: Bobby <table>s\ndefault: Bobby <table>s\n')

    def test_default_on(self):
        loader = DictLoader(self.templates, autoescape='xhtml_escape')
        name = 'Bobby <table>s'
        self.assertEqual(loader.load('escaped.html').generate(name=name), b'Bobby &lt;table&gt;s')
        self.assertEqual(loader.load('unescaped.html').generate(name=name), b'Bobby <table>s')
        self.assertEqual(loader.load('default.html').generate(name=name), b'Bobby &lt;table&gt;s')
        self.assertEqual(loader.load('include.html').generate(name=name), b'escaped: Bobby &lt;table&gt;s\nunescaped: Bobby <table>s\ndefault: Bobby &lt;table&gt;s\n')

    def test_unextended_block(self):
        loader = DictLoader(self.templates)
        name = '<script>'
        self.assertEqual(loader.load('escaped_block.html').generate(name=name), b'base: &lt;script&gt;')
        self.assertEqual(loader.load('unescaped_block.html').generate(name=name), b'base: <script>')

    def test_extended_block(self):
        loader = DictLoader(self.templates)

        def render(name):
            return loader.load(name).generate(name='<script>')
        self.assertEqual(render('escaped_extends_unescaped.html'), b'base: <script>')
        self.assertEqual(render('escaped_overrides_unescaped.html'), b'extended: &lt;script&gt;')
        self.assertEqual(render('unescaped_extends_escaped.html'), b'base: &lt;script&gt;')
        self.assertEqual(render('unescaped_overrides_escaped.html'), b'extended: <script>')

    def test_raw_expression(self):
        loader = DictLoader(self.templates)

        def render(name):
            return loader.load(name).generate(name='<>&"')
        self.assertEqual(render('raw_expression.html'), b'expr: &lt;&gt;&amp;&quot;\nraw: <>&"')

    def test_custom_escape(self):
        loader = DictLoader({'foo.py': '{% autoescape py_escape %}s = {{ name }}\n'})

        def py_escape(s):
            self.assertEqual(type(s), bytes)
            return repr(native_str(s))

        def render(template, name):
            return loader.load(template).generate(py_escape=py_escape, name=name)
        self.assertEqual(render('foo.py', '<html>'), b"s = '<html>'\n")
        self.assertEqual(render('foo.py', "';sys.exit()"), b's = "\';sys.exit()"\n')
        self.assertEqual(render('foo.py', ['not a string']), b's = "[\'not a string\']"\n')

    def test_manual_minimize_whitespace(self):
        loader = DictLoader({'foo.txt': '{% for i in items\n  %}{% if i > 0 %}, {% end %}{#\n  #}{{i\n  }}{% end\n%}'})
        self.assertEqual(loader.load('foo.txt').generate(items=range(5)), b'0, 1, 2, 3, 4')

    def test_whitespace_by_filename(self):
        loader = DictLoader({'foo.html': '   \n\t\n asdf\t   ', 'bar.js': ' \n\n\n\t qwer     ', 'baz.txt': '\t    zxcv\n\n', 'include.html': '  {% include baz.txt %} \n ', 'include.txt': '\t\t{% include foo.html %}    '})
        self.assertEqual(loader.load('foo.html').generate(), b'\nasdf ')
        self.assertEqual(loader.load('bar.js').generate(), b'\nqwer ')
        self.assertEqual(loader.load('baz.txt').generate(), b'\t    zxcv\n\n')
        self.assertEqual(loader.load('include.html').generate(), b' \t    zxcv\n\n\n')
        self.assertEqual(loader.load('include.txt').generate(), b'\t\t\nasdf     ')

    def test_whitespace_by_loader(self):
        templates = {'foo.html': '\t\tfoo\n\n', 'bar.txt': '\t\tbar\n\n'}
        loader = DictLoader(templates, whitespace='all')
        self.assertEqual(loader.load('foo.html').generate(), b'\t\tfoo\n\n')
        self.assertEqual(loader.load('bar.txt').generate(), b'\t\tbar\n\n')
        loader = DictLoader(templates, whitespace='single')
        self.assertEqual(loader.load('foo.html').generate(), b' foo\n')
        self.assertEqual(loader.load('bar.txt').generate(), b' bar\n')
        loader = DictLoader(templates, whitespace='oneline')
        self.assertEqual(loader.load('foo.html').generate(), b' foo ')
        self.assertEqual(loader.load('bar.txt').generate(), b' bar ')

    def test_whitespace_directive(self):
        loader = DictLoader({'foo.html': '{% whitespace oneline %}\n    {% for i in range(3) %}\n        {{ i }}\n    {% end %}\n{% whitespace all %}\n    pre\tformatted\n'})
        self.assertEqual(loader.load('foo.html').generate(), b'  0  1  2  \n    pre\tformatted\n')