import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
class MarkupTestCase(unittest.TestCase):

    def test_new_with_encoding(self):
        markup = Markup(u'Döner'.encode('utf-8'), encoding='utf-8')
        self.assertEqual('<Markup %r>' % u'Döner', repr(markup))

    def test_repr(self):
        markup = Markup('foo')
        expected_foo = "u'foo'" if IS_PYTHON2 else "'foo'"
        self.assertEqual('<Markup %s>' % expected_foo, repr(markup))

    def test_escape(self):
        markup = escape('<b>"&"</b>')
        assert type(markup) is Markup
        self.assertEqual('&lt;b&gt;&#34;&amp;&#34;&lt;/b&gt;', markup)

    def test_escape_noquotes(self):
        markup = escape('<b>"&"</b>', quotes=False)
        assert type(markup) is Markup
        self.assertEqual('&lt;b&gt;"&amp;"&lt;/b&gt;', markup)

    def test_unescape_markup(self):
        string = '<b>"&"</b>'
        markup = Markup.escape(string)
        assert type(markup) is Markup
        self.assertEqual(string, unescape(markup))

    def test_Markup_escape_None_noquotes(self):
        markup = Markup.escape(None, False)
        assert type(markup) is Markup
        self.assertEqual('', markup)

    def test_add_str(self):
        markup = Markup('<b>foo</b>') + '<br/>'
        assert type(markup) is Markup
        self.assertEqual('<b>foo</b>&lt;br/&gt;', markup)

    def test_add_markup(self):
        markup = Markup('<b>foo</b>') + Markup('<br/>')
        assert type(markup) is Markup
        self.assertEqual('<b>foo</b><br/>', markup)

    def test_add_reverse(self):
        markup = '<br/>' + Markup('<b>bar</b>')
        assert type(markup) is Markup
        self.assertEqual('&lt;br/&gt;<b>bar</b>', markup)

    def test_mod(self):
        markup = Markup('<b>%s</b>') % '&'
        assert type(markup) is Markup
        self.assertEqual('<b>&amp;</b>', markup)

    def test_mod_multi(self):
        markup = Markup('<b>%s</b> %s') % ('&', 'boo')
        assert type(markup) is Markup
        self.assertEqual('<b>&amp;</b> boo', markup)

    def test_mod_mapping(self):
        markup = Markup('<b>%(foo)s</b>') % {'foo': '&'}
        assert type(markup) is Markup
        self.assertEqual('<b>&amp;</b>', markup)

    def test_mod_noescape(self):
        markup = Markup('<b>%(amp)s</b>') % {'amp': Markup('&amp;')}
        assert type(markup) is Markup
        self.assertEqual('<b>&amp;</b>', markup)

    def test_mul(self):
        markup = Markup('<b>foo</b>') * 2
        assert type(markup) is Markup
        self.assertEqual('<b>foo</b><b>foo</b>', markup)

    def test_mul_reverse(self):
        markup = 2 * Markup('<b>foo</b>')
        assert type(markup) is Markup
        self.assertEqual('<b>foo</b><b>foo</b>', markup)

    def test_join(self):
        markup = Markup('<br />').join(['foo', '<bar />', Markup('<baz />')])
        assert type(markup) is Markup
        self.assertEqual('foo<br />&lt;bar /&gt;<br /><baz />', markup)

    def test_join_over_iter(self):
        items = ['foo', '<bar />', Markup('<baz />')]
        markup = Markup('<br />').join((i for i in items))
        self.assertEqual('foo<br />&lt;bar /&gt;<br /><baz />', markup)

    def test_stripentities_all(self):
        markup = Markup('&amp; &#106;').stripentities()
        assert type(markup) is Markup
        self.assertEqual('& j', markup)

    def test_stripentities_keepxml(self):
        markup = Markup('&amp; &#106;').stripentities(keepxmlentities=True)
        assert type(markup) is Markup
        self.assertEqual('&amp; j', markup)

    def test_striptags_empty(self):
        markup = Markup('<br />').striptags()
        assert type(markup) is Markup
        self.assertEqual('', markup)

    def test_striptags_mid(self):
        markup = Markup('<a href="#">fo<br />o</a>').striptags()
        assert type(markup) is Markup
        self.assertEqual('foo', markup)

    def test_pickle(self):
        markup = Markup('foo')
        buf = BytesIO()
        pickle.dump(markup, buf, 2)
        buf.seek(0)
        expected_foo = "u'foo'" if IS_PYTHON2 else "'foo'"
        self.assertEqual('<Markup %s>' % expected_foo, repr(pickle.load(buf)))