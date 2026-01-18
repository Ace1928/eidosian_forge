from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
class ContextDirectiveTestCase(unittest.TestCase):

    def test_extract_msgcontext(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/"\n                                xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:ctxt="foo">Foo, bar.</p>\n          <p>Foo, bar.</p>\n        </html>')
        results = list(extract(buf, ['_'], [], {}))
        self.assertEqual((3, 'pgettext', ('foo', 'Foo, bar.'), []), results[0])
        self.assertEqual((4, None, 'Foo, bar.', []), results[1])

    def test_translate_msgcontext(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:ctxt="foo">Foo, bar.</p>\n          <p>Foo, bar.</p>\n        </html>')
        translations = {('foo', 'Foo, bar.'): 'Fooo! Barrr!', 'Foo, bar.': 'Foo --- bar.'}
        translator = Translator(DummyTranslations(translations))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Fooo! Barrr!</p>\n          <p>Foo --- bar.</p>\n        </html>', tmpl.generate().render())

    def test_translate_msgcontext_with_domain(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:domain="bar" i18n:ctxt="foo">Foo, bar. <span>foo</span></p>\n          <p>Foo, bar.</p>\n        </html>')
        translations = DummyTranslations({('foo', 'Foo, bar.'): 'Fooo! Barrr!', 'Foo, bar.': 'Foo --- bar.'})
        translations.add_domain('bar', {('foo', 'foo'): 'BARRR', ('foo', 'Foo, bar.'): 'Bar, bar.'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Bar, bar. <span>BARRR</span></p>\n          <p>Foo --- bar.</p>\n        </html>', tmpl.generate().render())

    def test_translate_msgcontext_with_plurals(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:ctxt name="foo">\n          <p i18n:choose="num; num">\n            <span i18n:singular="">There is ${num} bar</span>\n            <span i18n:plural="">There are ${num} bars</span>\n          </p>\n        </i18n:ctxt>\n        </html>')
        translations = DummyTranslations({('foo', 'There is %(num)s bar', 0): 'Hay %(num)s barre', ('foo', 'There is %(num)s bar', 1): 'Hay %(num)s barres'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>\n            <span>Hay 1 barre</span>\n          </p>\n        </html>', tmpl.generate(num=1).render())
        self.assertEqual('<html>\n          <p>\n            <span>Hay 2 barres</span>\n          </p>\n        </html>', tmpl.generate(num=2).render())

    def test_translate_context_with_msg(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <p i18n:ctxt="foo" i18n:msg="num">\n          Foo <span>There is ${num} bar</span> Bar\n        </p>\n        </html>')
        translations = DummyTranslations({('foo', 'Foo [1:There is %(num)s bar] Bar'): 'Voh [1:Hay %(num)s barre] Barre'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n        <p>Voh <span>Hay 1 barre</span> Barre</p>\n        </html>', tmpl.generate(num=1).render())