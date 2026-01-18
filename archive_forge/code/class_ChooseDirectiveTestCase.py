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
class ChooseDirectiveTestCase(unittest.TestCase):

    def test_translate_i18n_choose_as_attribute(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="one">\n            <p i18n:singular="">FooBar</p>\n            <p i18n:plural="">FooBars</p>\n          </div>\n          <div i18n:choose="two">\n            <p i18n:singular="">FooBar</p>\n            <p i18n:plural="">FooBars</p>\n          </div>\n        </html>')
        translations = DummyTranslations()
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>FooBar</p>\n          </div>\n          <div>\n            <p>FooBars</p>\n          </div>\n        </html>', tmpl.generate(one=1, two=2).render())

    def test_translate_i18n_choose_as_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <p i18n:singular="">FooBar</p>\n          <p i18n:plural="">FooBars</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="">FooBar</p>\n          <p i18n:plural="">FooBars</p>\n        </i18n:choose>\n        </html>')
        translations = DummyTranslations()
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>FooBars</p>\n          <p>FooBar</p>\n        </html>', tmpl.generate(one=1, two=2).render())

    def test_translate_i18n_choose_as_directive_singular_and_plural_with_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <p i18n:singular="" py:strip="">FooBar Singular with Strip</p>\n          <p i18n:plural="">FooBars Plural without Strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="two">\n          <p i18n:singular="">FooBar singular without strip</p>\n          <p i18n:plural="" py:strip="">FooBars plural with strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="">FooBar singular without strip</p>\n          <p i18n:plural="" py:strip="">FooBars plural with strip</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="" py:strip="">FooBar singular with strip</p>\n          <p i18n:plural="">FooBars plural without strip</p>\n        </i18n:choose>\n        </html>')
        translations = DummyTranslations()
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>FooBars Plural without Strip</p>\n          FooBars plural with strip\n          <p>FooBar singular without strip</p>\n          FooBar singular with strip\n        </html>', tmpl.generate(one=1, two=2).render())

    def test_translate_i18n_choose_plural_singular_as_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <i18n:singular>FooBar</i18n:singular>\n          <i18n:plural>FooBars</i18n:plural>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <i18n:singular>FooBar</i18n:singular>\n          <i18n:plural>FooBars</i18n:plural>\n        </i18n:choose>\n        </html>')
        translations = DummyTranslations({('FooBar', 0): 'FuBar', ('FooBars', 1): 'FuBars', 'FooBar': 'FuBar', 'FooBars': 'FuBars'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          FuBars\n          FuBar\n        </html>', tmpl.generate(one=1, two=2).render())

    def test_translate_i18n_choose_as_attribute_with_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>Vohs John Doe</p>\n          </div>\n        </html>', tmpl.generate(two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_as_attribute_with_params_and_domain_as_param(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n"\n            i18n:domain="foo">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations()
        translations.add_domain('foo', {('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>Vohs John Doe</p>\n          </div>\n        </html>', tmpl.generate(two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_as_directive_with_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        <i18n:choose numeral="one" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
        translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Vohs John Doe</p>\n          <p>Voh John Doe</p>\n        </html>', tmpl.generate(one=1, two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_as_directive_with_params_and_domain_as_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:domain name="foo">\n        <i18n:choose numeral="two" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </i18n:domain>\n        <i18n:choose numeral="one" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
        translations = DummyTranslations()
        translations.add_domain('foo', {('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Vohs John Doe</p>\n          <p>Foo John Doe</p>\n        </html>', tmpl.generate(one=1, two=2, fname='John', lname='Doe').render())

    def test_extract_i18n_choose_as_attribute(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="one">\n            <p i18n:singular="">FooBar</p>\n            <p i18n:plural="">FooBars</p>\n          </div>\n          <div i18n:choose="two">\n            <p i18n:singular="">FooBar</p>\n            <p i18n:plural="">FooBars</p>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual((3, 'ngettext', ('FooBar', 'FooBars'), []), messages[0])
        self.assertEqual((7, 'ngettext', ('FooBar', 'FooBars'), []), messages[1])

    def test_extract_i18n_choose_as_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two">\n          <p i18n:singular="">FooBar</p>\n          <p i18n:plural="">FooBars</p>\n        </i18n:choose>\n        <i18n:choose numeral="one">\n          <p i18n:singular="">FooBar</p>\n          <p i18n:plural="">FooBars</p>\n        </i18n:choose>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual((3, 'ngettext', ('FooBar', 'FooBars'), []), messages[0])
        self.assertEqual((7, 'ngettext', ('FooBar', 'FooBars'), []), messages[1])

    def test_extract_i18n_choose_as_attribute_with_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[0])

    def test_extract_i18n_choose_as_attribute_with_params_and_domain_as_param(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n"\n            i18n:domain="foo">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((4, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[0])

    def test_extract_i18n_choose_as_directive_with_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        <i18n:choose numeral="one" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual((3, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[0])
        self.assertEqual((7, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[1])

    def test_extract_i18n_choose_as_directive_with_params_and_domain_as_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:domain name="foo">\n        <i18n:choose numeral="two" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </i18n:domain>\n        <i18n:choose numeral="one" params="fname, lname">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual((4, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[0])
        self.assertEqual((9, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), []), messages[1])

    def test_extract_i18n_choose_as_attribute_with_params_and_comment(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname" i18n:comment="As in Foo Bar">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), ['As in Foo Bar']), messages[0])

    def test_extract_i18n_choose_as_directive_with_params_and_comment(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n        <i18n:choose numeral="two" params="fname, lname" i18n:comment="As in Foo Bar">\n          <p i18n:singular="">Foo ${fname} ${lname}</p>\n          <p i18n:plural="">Foos ${fname} ${lname}</p>\n        </i18n:choose>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, 'ngettext', ('Foo %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s'), ['As in Foo Bar']), messages[0])

    def test_extract_i18n_choose_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:choose="num; num" title="Things">\n            <i18n:singular>\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </i18n:singular>\n            <i18n:plural>\n              There are <a href="$link" title="View things">${num} things</a>.\n            </i18n:plural>\n          </p>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(4, len(messages))
        self.assertEqual((3, None, 'Things', []), messages[0])
        self.assertEqual((5, None, 'View thing', []), messages[1])
        self.assertEqual((8, None, 'View things', []), messages[2])
        self.assertEqual((3, 'ngettext', ('There is [1:%(num)s thing].', 'There are [1:%(num)s things].'), []), messages[3])

    def test_translate_i18n_choose_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:choose="num; num" title="Things">\n            <i18n:singular>\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </i18n:singular>\n            <i18n:plural>\n              There are <a href="$link" title="View things">${num} things</a>.\n            </i18n:plural>\n          </p>\n        </html>')
        translations = DummyTranslations({'Things': 'Sachen', 'View thing': 'Sache betrachten', 'View things': 'Sachen betrachten', ('There is [1:%(num)s thing].', 0): 'Da ist [1:%(num)s Sache].', ('There is [1:%(num)s thing].', 1): 'Da sind [1:%(num)s Sachen].'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p title="Sachen">\n            Da ist <a href="/things" title="Sache betrachten">1 Sache</a>.\n          </p>\n        </html>', tmpl.generate(link='/things', num=1).render(encoding=None))
        self.assertEqual(u'<html>\n          <p title="Sachen">\n            Da sind <a href="/things" title="Sachen betrachten">3 Sachen</a>.\n          </p>\n        </html>', tmpl.generate(link='/things', num=3).render(encoding=None))

    def test_extract_i18n_choose_as_element_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:choose numeral="num" params="num">\n            <p i18n:singular="" title="Things">\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </p>\n            <p i18n:plural="" title="Things">\n              There are <a href="$link" title="View things">${num} things</a>.\n            </p>\n          </i18n:choose>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(5, len(messages))
        self.assertEqual((4, None, 'Things', []), messages[0])
        self.assertEqual((5, None, 'View thing', []), messages[1])
        self.assertEqual((7, None, 'Things', []), messages[2])
        self.assertEqual((8, None, 'View things', []), messages[3])
        self.assertEqual((3, 'ngettext', ('There is [1:%(num)s thing].', 'There are [1:%(num)s things].'), []), messages[4])

    def test_translate_i18n_choose_as_element_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:choose numeral="num" params="num">\n            <p i18n:singular="" title="Things">\n              There is <a href="$link" title="View thing">${num} thing</a>.\n            </p>\n            <p i18n:plural="" title="Things">\n              There are <a href="$link" title="View things">${num} things</a>.\n            </p>\n          </i18n:choose>\n        </html>')
        translations = DummyTranslations({'Things': 'Sachen', 'View thing': 'Sache betrachten', 'View things': 'Sachen betrachten', ('There is [1:%(num)s thing].', 0): 'Da ist [1:%(num)s Sache].', ('There is [1:%(num)s thing].', 1): 'Da sind [1:%(num)s Sachen].'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n            <p title="Sachen">Da ist <a href="/things" title="Sache betrachten">1 Sache</a>.</p>\n        </html>', tmpl.generate(link='/things', num=1).render(encoding=None))
        self.assertEqual(u'<html>\n            <p title="Sachen">Da sind <a href="/things" title="Sachen betrachten">3 Sachen</a>.</p>\n        </html>', tmpl.generate(link='/things', num=3).render(encoding=None))

    def test_translate_i18n_choose_and_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>Vohs John Doe</p>\n          </div>\n        </html>', tmpl.generate(two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_and_domain_and_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n"\n            i18n:domain="foo">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations()
        translations.add_domain('foo', {('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>Vohs John Doe</p>\n          </div>\n        </html>', tmpl.generate(two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_and_singular_with_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="" py:strip="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n          <div i18n:choose="one; fname, lname">\n            <p i18n:singular="" py:strip="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            <p>Vohs John Doe</p>\n          </div>\n          <div>\n            Voh John Doe\n          </div>\n        </html>', tmpl.generate(one=1, two=2, fname='John', lname='Doe').render())

    def test_translate_i18n_choose_and_plural_with_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="two; fname, lname">\n            <p i18n:singular="" py:strip="">Foo $fname $lname</p>\n            <p i18n:plural="">Foos $fname $lname</p>\n          </div>\n        </html>')
        translations = DummyTranslations({('Foo %(fname)s %(lname)s', 0): 'Voh %(fname)s %(lname)s', ('Foo %(fname)s %(lname)s', 1): 'Vohs %(fname)s %(lname)s', 'Foo %(fname)s %(lname)s': 'Voh %(fname)s %(lname)s', 'Foos %(fname)s %(lname)s': 'Vohs %(fname)s %(lname)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div>\n            Voh John Doe\n          </div>\n        </html>', tmpl.generate(two=1, fname='John', lname='Doe').render())

    def test_extract_i18n_choose_as_attribute_and_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:choose="one" py:strip="">\n            <p i18n:singular="" py:strip="">FooBar</p>\n            <p i18n:plural="" py:strip="">FooBars</p>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, 'ngettext', ('FooBar', 'FooBars'), []), messages[0])