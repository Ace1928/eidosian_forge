import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class MatchDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:match` template directive."""

    def test_with_strip(self):
        """
        Verify that a match template can produce the same kind of element that
        it matched without entering an infinite recursion.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:match="elem" py:strip="">\n            <div class="elem">${select(\'text()\')}</div>\n          </elem>\n          <elem>Hey Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n            <div class="elem">Hey Joe</div>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_without_strip(self):
        """
        Verify that a match template can produce the same kind of element that
        it matched without entering an infinite recursion.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:match="elem">\n            <div class="elem">${select(\'text()\')}</div>\n          </elem>\n          <elem>Hey Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n          <elem>\n            <div class="elem">Hey Joe</div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_as_element(self):
        """
        Verify that the directive can also be used as an element.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="elem">\n            <div class="elem">${select(\'text()\')}</div>\n          </py:match>\n          <elem>Hey Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n            <div class="elem">Hey Joe</div>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_recursive_match_1(self):
        """
        Match directives are applied recursively, meaning that they are also
        applied to any content they may have produced themselves:
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <elem py:match="elem">\n            <div class="elem">\n              ${select(\'*\')}\n            </div>\n          </elem>\n          <elem>\n            <subelem>\n              <elem/>\n            </subelem>\n          </elem>\n        </doc>')
        self.assertEqual('<doc>\n          <elem>\n            <div class="elem">\n              <subelem>\n              <elem>\n            <div class="elem">\n            </div>\n          </elem>\n            </subelem>\n            </div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_recursive_match_2(self):
        """
        When two or more match templates match the same element and also
        themselves output the element they match, avoiding recursion is even
        more complex, but should work.
        """
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body">\n            <div id="header"/>\n            ${select(\'*\')}\n          </body>\n          <body py:match="body">\n            ${select(\'*\')}\n            <div id="footer"/>\n          </body>\n          <body>\n            <h1>Foo</h1>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <body>\n            <div id="header"/><h1>Foo</h1>\n            <div id="footer"/>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_recursive_match_3(self):
        tmpl = MarkupTemplate('<test xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="b[@type=\'bullet\']">\n            <bullet>${select(\'*|text()\')}</bullet>\n          </py:match>\n          <py:match path="group[@type=\'bullet\']">\n            <ul>${select(\'*\')}</ul>\n          </py:match>\n          <py:match path="b">\n            <generic>${select(\'*|text()\')}</generic>\n          </py:match>\n\n          <b>\n            <group type="bullet">\n              <b type="bullet">1</b>\n              <b type="bullet">2</b>\n            </group>\n          </b>\n        </test>\n        ')
        self.assertEqual('<test>\n            <generic>\n            <ul><bullet>1</bullet><bullet>2</bullet></ul>\n          </generic>\n        </test>', tmpl.generate().render(encoding=None))

    def test_not_match_self(self):
        """
        See http://genshi.edgewall.org/ticket/77
        """
        tmpl = MarkupTemplate('<html xmlns="http://www.w3.org/1999/xhtml"\n              xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body" py:content="select(\'*\')" />\n          <h1 py:match="h1">\n            ${select(\'text()\')}\n            Goodbye!\n          </h1>\n          <body>\n            <h1>Hello!</h1>\n          </body>\n        </html>')
        self.assertEqual('<html xmlns="http://www.w3.org/1999/xhtml">\n          <body><h1>\n            Hello!\n            Goodbye!\n          </h1></body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_select_text_in_element(self):
        """
        See http://genshi.edgewall.org/ticket/77#comment:1
        """
        tmpl = MarkupTemplate('<html xmlns="http://www.w3.org/1999/xhtml"\n              xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body" py:content="select(\'*\')" />\n          <h1 py:match="h1">\n            <text>\n              ${select(\'text()\')}\n            </text>\n            Goodbye!\n          </h1>\n          <body>\n            <h1>Hello!</h1>\n          </body>\n        </html>')
        self.assertEqual('<html xmlns="http://www.w3.org/1999/xhtml">\n          <body><h1>\n            <text>\n              Hello!\n            </text>\n            Goodbye!\n          </h1></body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_select_all_attrs(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="elem" py:attrs="select(\'@*\')">\n            ${select(\'text()\')}\n          </div>\n          <elem id="joe">Hey Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n          <div id="joe">\n            Hey Joe\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_select_all_attrs_empty(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="elem" py:attrs="select(\'@*\')">\n            ${select(\'text()\')}\n          </div>\n          <elem>Hey Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n          <div>\n            Hey Joe\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_select_all_attrs_in_body(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="elem">\n            Hey ${select(\'text()\')} ${select(\'@*\')}\n          </div>\n          <elem title="Cool">Joe</elem>\n        </doc>')
        self.assertEqual('<doc>\n          <div>\n            Hey Joe Cool\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_def_in_match(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="maketitle(test)"><b py:replace="test" /></py:def>\n          <head py:match="head">${select(\'*\')}</head>\n          <head><title>${maketitle(True)}</title></head>\n        </doc>')
        self.assertEqual('<doc>\n          <head><title>True</title></head>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_match_with_xpath_variable(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:match="*[name()=$tagname]">\n            Hello ${select(\'@name\')}\n          </span>\n          <greeting name="Dude"/>\n        </div>')
        self.assertEqual('<div>\n          <span>\n            Hello Dude\n          </span>\n        </div>', tmpl.generate(tagname='greeting').render(encoding=None))
        self.assertEqual('<div>\n          <greeting name="Dude"/>\n        </div>', tmpl.generate(tagname='sayhello').render(encoding=None))

    def test_content_directive_in_match(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="foo">I said <q py:content="select(\'text()\')">something</q>.</div>\n          <foo>bar</foo>\n        </html>')
        self.assertEqual('<html>\n          <div>I said <q>bar</q>.</div>\n        </html>', tmpl.generate().render(encoding=None))

    def test_cascaded_matches(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body">${select(\'*\')}</body>\n          <head py:match="head">${select(\'title\')}</head>\n          <body py:match="body">${select(\'*\')}<hr /></body>\n          <head><title>Welcome to Markup</title></head>\n          <body><h2>Are you ready to mark up?</h2></body>\n        </html>')
        self.assertEqual('<html>\n          <head><title>Welcome to Markup</title></head>\n          <body><h2>Are you ready to mark up?</h2><hr/></body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_multiple_matches(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <input py:match="form//input" py:attrs="select(\'@*\')"\n                 value="${values[str(select(\'@name\'))]}" />\n          <form><p py:for="field in fields">\n            <label>${field.capitalize()}</label>\n            <input type="text" name="${field}" />\n          </p></form>\n        </html>')
        fields = ['hello_%s' % i for i in range(5)]
        values = dict([('hello_%s' % i, i) for i in range(5)])
        self.assertEqual('<html>\n          <form><p>\n            <label>Hello_0</label>\n            <input value="0" type="text" name="hello_0"/>\n          </p><p>\n            <label>Hello_1</label>\n            <input value="1" type="text" name="hello_1"/>\n          </p><p>\n            <label>Hello_2</label>\n            <input value="2" type="text" name="hello_2"/>\n          </p><p>\n            <label>Hello_3</label>\n            <input value="3" type="text" name="hello_3"/>\n          </p><p>\n            <label>Hello_4</label>\n            <input value="4" type="text" name="hello_4"/>\n          </p></form>\n        </html>', tmpl.generate(fields=fields, values=values).render(encoding=None))

    def test_namespace_context(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n                                       xmlns:x="http://www.example.org/">\n          <div py:match="x:foo">Foo</div>\n          <foo xmlns="http://www.example.org/"/>\n        </html>')
        self.assertEqual('<html xmlns:x="http://www.example.org/">\n          <div>Foo</div>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_with_position_predicate(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p py:match="body/p[1]" class="first">${select(\'*|text()\')}</p>\n          <body>\n            <p>Foo</p>\n            <p>Bar</p>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <body>\n            <p class="first">Foo</p>\n            <p>Bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_with_closure(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p py:match="body//p" class="para">${select(\'*|text()\')}</p>\n          <body>\n            <p>Foo</p>\n            <div><p>Bar</p></div>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <body>\n            <p class="para">Foo</p>\n            <div><p class="para">Bar</p></div>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_without_closure(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p py:match="body/p" class="para">${select(\'*|text()\')}</p>\n          <body>\n            <p>Foo</p>\n            <div><p>Bar</p></div>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <body>\n            <p class="para">Foo</p>\n            <div><p>Bar</p></div>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_with_once_attribute(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body" once="true"><body>\n            <div id="wrap">\n              ${select("*")}\n            </div>\n          </body></py:match>\n          <body>\n            <p>Foo</p>\n          </body>\n          <body>\n            <p>Bar</p>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <body>\n            <div id="wrap">\n              <p>Foo</p>\n            </div>\n          </body>\n          <body>\n            <p>Bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_with_recursive_attribute(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="elem" recursive="false"><elem>\n            <div class="elem">\n              ${select(\'*\')}\n            </div>\n          </elem></py:match>\n          <elem>\n            <subelem>\n              <elem/>\n            </subelem>\n          </elem>\n        </doc>')
        self.assertEqual('<doc>\n          <elem>\n            <div class="elem">\n              <subelem>\n              <elem/>\n            </subelem>\n            </div>\n          </elem>\n        </doc>', tmpl.generate().render(encoding=None))

    def test_triple_match_produces_no_duplicate_items(self):
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="div[@id=\'content\']" py:attrs="select(\'@*\')" once="true">\n            <ul id="tabbed_pane" />\n            ${select(\'*\')}\n          </div>\n\n          <body py:match="body" once="true" buffer="false">\n            ${select(\'*|text()\')}\n          </body>\n          <body py:match="body" once="true" buffer="false">\n              ${select(\'*|text()\')}\n          </body>\n\n          <body>\n            <div id="content">\n              <h1>Ticket X</h1>\n            </div>\n          </body>\n        </doc>')
        output = tmpl.generate().render('xhtml', doctype='xhtml')
        matches = re.findall('tabbed_pane', output)
        self.assertNotEqual(None, matches)
        self.assertEqual(1, len(matches))

    def test_match_multiple_times1(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body[@id=\'content\']/h2" />\n          <head py:match="head" />\n          <head py:match="head" />\n          <head />\n          <body />\n        </html>')
        self.assertEqual('<html>\n          <head/>\n          <body/>\n        </html>', tmpl.generate().render())

    def test_match_multiple_times2(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/div[@id=\'properties\']" />\n          <head py:match="head" />\n          <head py:match="head" />\n          <head/>\n          <body>\n            <div id="properties">Foo</div>\n          </body>\n        </html>')
        self.assertEqual('<html>\n          <head/>\n          <body>\n          </body>\n        </html>', tmpl.generate().render())

    def test_match_multiple_times3(self):
        tmpl = MarkupTemplate('<?xml version="1.0"?>\n          <root xmlns:py="http://genshi.edgewall.org/">\n            <py:match path="foo/bar">\n              <zzzzz/>\n            </py:match>\n            <foo>\n              <bar/>\n              <bar/>\n            </foo>\n            <bar/>\n          </root>')
        self.assertEqual('<?xml version="1.0"?>\n<root>\n            <foo>\n              <zzzzz/>\n              <zzzzz/>\n            </foo>\n            <bar/>\n          </root>', tmpl.generate().render())