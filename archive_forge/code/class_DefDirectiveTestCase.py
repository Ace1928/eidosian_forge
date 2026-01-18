import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class DefDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:def` template directive."""

    def test_function_with_strip(self):
        """
        Verify that a named template function with a strip directive actually
        strips of the outer element.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="echo(what)" py:strip="">\n            <b>${what}</b>\n          </div>\n          ${echo(\'foo\')}\n        </doc>')
        self.assertEqual('<doc>\n            <b>foo</b>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_exec_in_replace(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <p py:def="echo(greeting, name=\'world\')" class="message">\n            ${greeting}, ${name}!\n          </p>\n          <div py:replace="echo(\'hello\')"></div>\n        </div>')
        self.assertEqual('<div>\n          <p class="message">\n            hello, world!\n          </p>\n        </div>', tmpl.generate().render(encoding=None))

    def test_as_element(self):
        """
        Verify that the directive can also be used as an element.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="echo(what)">\n            <b>${what}</b>\n          </py:def>\n          ${echo(\'foo\')}\n        </doc>')
        self.assertEqual('<doc>\n            <b>foo</b>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_nested_defs(self):
        """
        Verify that a template function defined inside a conditional block can
        be called from outside that block.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:if test="semantic">\n            <strong py:def="echo(what)">${what}</strong>\n          </py:if>\n          <py:if test="not semantic">\n            <b py:def="echo(what)">${what}</b>\n          </py:if>\n          ${echo(\'foo\')}\n        </doc>')
        self.assertEqual('<doc>\n          <strong>foo</strong>\n        </doc>', tmpl.generate(semantic=True).render(encoding=None))

    def test_function_with_default_arg(self):
        """
        Verify that keyword arguments work with `py:def` directives.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <b py:def="echo(what, bold=False)" py:strip="not bold">${what}</b>\n          ${echo(\'foo\')}\n        </doc>')
        self.assertEqual('<doc>\n          foo\n        </doc>', tmpl.generate().render(encoding=None))

    def test_invocation_in_attribute(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="echo(what)">${what or \'something\'}</py:def>\n          <p class="${echo(\'foo\')}">bar</p>\n        </doc>')
        self.assertEqual('<doc>\n          <p class="foo">bar</p>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_invocation_in_attribute_none(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="echo()">${None}</py:def>\n          <p class="${echo()}">bar</p>\n        </doc>')
        self.assertEqual('<doc>\n          <p>bar</p>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_function_raising_typeerror(self):

        def badfunc():
            raise TypeError
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="dobadfunc()">\n            ${badfunc()}\n          </div>\n          <div py:content="dobadfunc()"/>\n        </html>')
        self.assertRaises(TypeError, list, tmpl.generate(badfunc=badfunc))

    def test_def_in_matched(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <head py:match="head">${select(\'*\')}</head>\n          <head>\n            <py:def function="maketitle(test)"><b py:replace="test" /></py:def>\n            <title>${maketitle(True)}</title>\n          </head>\n        </doc>')
        self.assertEqual('<doc>\n          <head><title>True</title></head>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_in_text_template(self):
        """
        Verify that the directive works as expected in a text template.
        """
        tmpl = TextTemplate("\n          #def echo(greeting, name='world')\n            ${greeting}, ${name}!\n          #end\n          ${echo('Hi', name='you')}\n        ")
        self.assertEqual('\n                      Hi, you!\n\n        ', tmpl.generate().render(encoding=None))

    def test_function_with_star_args(self):
        """
        Verify that a named template function using "star arguments" works as
        expected.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="f(*args, **kwargs)">\n            ${repr(args)}\n            ${repr(sorted(kwargs.items()))}\n          </div>\n          ${f(1, 2, a=3, b=4)}\n        </doc>')
        self.assertEqual("<doc>\n          <div>\n            [1, 2]\n            [('a', 3), ('b', 4)]\n          </div>\n        </doc>", tmpl.generate().render(encoding=None))