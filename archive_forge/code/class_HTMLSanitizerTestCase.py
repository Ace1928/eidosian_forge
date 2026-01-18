import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
class HTMLSanitizerTestCase(unittest.TestCase):

    def assert_parse_error_or_equal(self, expected, exploit, allow_strip=False):
        try:
            html = HTML(exploit)
        except ParseError:
            return
        sanitized_html = (html | HTMLSanitizer()).render()
        if not sanitized_html and allow_strip:
            return
        self.assertEqual(expected, sanitized_html)

    def test_sanitize_unchanged(self):
        html = HTML(u'<a href="#">fo<br />o</a>')
        self.assertEqual('<a href="#">fo<br/>o</a>', (html | HTMLSanitizer()).render())
        html = HTML(u'<a href="#with:colon">foo</a>')
        self.assertEqual('<a href="#with:colon">foo</a>', (html | HTMLSanitizer()).render())

    def test_sanitize_escape_text(self):
        html = HTML(u'<a href="#">fo&amp;</a>')
        self.assertEqual('<a href="#">fo&amp;</a>', (html | HTMLSanitizer()).render())
        html = HTML(u'<a href="#">&lt;foo&gt;</a>')
        self.assertEqual('<a href="#">&lt;foo&gt;</a>', (html | HTMLSanitizer()).render())

    def test_sanitize_entityref_text(self):
        html = HTML(u'<a href="#">fo&ouml;</a>')
        self.assertEqual(u'<a href="#">foö</a>', (html | HTMLSanitizer()).render(encoding=None))

    def test_sanitize_escape_attr(self):
        html = HTML(u'<div title="&lt;foo&gt;"></div>')
        self.assertEqual('<div title="&lt;foo&gt;"/>', (html | HTMLSanitizer()).render())

    def test_sanitize_close_empty_tag(self):
        html = HTML(u'<a href="#">fo<br>o</a>')
        self.assertEqual('<a href="#">fo<br/>o</a>', (html | HTMLSanitizer()).render())

    def test_sanitize_invalid_entity(self):
        html = HTML(u'&junk;')
        self.assertEqual('&amp;junk;', (html | HTMLSanitizer()).render())

    def test_sanitize_remove_script_elem(self):
        html = HTML(u'<script>alert("Foo")</script>')
        self.assertEqual('', (html | HTMLSanitizer()).render())
        html = HTML(u'<SCRIPT SRC="http://example.com/"></SCRIPT>')
        self.assertEqual('', (html | HTMLSanitizer()).render())
        src = u'<SCR\x00IPT>alert("foo")</SCR\x00IPT>'
        self.assert_parse_error_or_equal('&lt;SCR\x00IPT&gt;alert("foo")', src, allow_strip=True)
        src = u'<SCRIPT&XYZ SRC="http://example.com/"></SCRIPT>'
        self.assert_parse_error_or_equal('&lt;SCRIPT&amp;XYZ; SRC="http://example.com/"&gt;', src, allow_strip=True)

    def test_sanitize_remove_onclick_attr(self):
        html = HTML(u'<div onclick=\'alert("foo")\' />')
        self.assertEqual('<div/>', (html | HTMLSanitizer()).render())

    def test_sanitize_remove_input_password(self):
        html = HTML(u'<form><input type="password" /></form>')
        self.assertEqual('<form/>', (html | HTMLSanitizer()).render())

    def test_sanitize_remove_comments(self):
        html = HTML(u'<div><!-- conditional comment crap --></div>')
        self.assertEqual('<div/>', (html | HTMLSanitizer()).render())

    def test_sanitize_remove_style_scripts(self):
        sanitizer = StyleSanitizer()
        html = HTML(u'<DIV STYLE=\'background: url(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: url(&#1;javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: url("javascript:alert(foo)")\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'width: expression(alert("foo"));\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'width: e/**/xpression(alert("foo"));\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: url(javascript:alert("foo"));color: #fff\'>')
        self.assertEqual('<div style="color: #fff"/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: \\75rl(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: \\000075rl(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: \\75 rl(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: \\000075 rl(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<DIV STYLE=\'background: \\000075\r\nrl(javascript:alert("foo"))\'>')
        self.assertEqual('<div/>', (html | sanitizer).render())

    def test_sanitize_remove_style_phishing(self):
        sanitizer = StyleSanitizer()
        html = HTML(u'<div style="position:absolute;top:0"></div>')
        self.assertEqual('<div style="top:0"/>', (html | sanitizer).render())
        html = HTML(u'<div style="margin:10px 20px"></div>')
        self.assertEqual('<div style="margin:10px 20px"/>', (html | sanitizer).render())
        html = HTML(u'<div style="margin:-1000px 0 0"></div>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<div style="margin-left:-2000px 0 0"></div>')
        self.assertEqual('<div/>', (html | sanitizer).render())
        html = HTML(u'<div style="margin-left:1em 1em 1em -4000px"></div>')
        self.assertEqual('<div/>', (html | sanitizer).render())

    def test_sanitize_remove_src_javascript(self):
        html = HTML(u'<img src=\'javascript:alert("foo")\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        html = HTML(u'<IMG SRC=\'JaVaScRiPt:alert("foo")\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        src = u'<IMG SRC=`javascript:alert("RSnake says, \'foo\'")`>'
        self.assert_parse_error_or_equal('<img/>', src)
        html = HTML(u'<IMG SRC=\'&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;alert("foo")\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        html = HTML(u'<IMG SRC=\'&#0000106&#0000097&#0000118&#0000097&#0000115&#0000099&#0000114&#0000105&#0000112&#0000116&#0000058alert("foo")\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        html = HTML(u'<IMG SRC=\'&#x6A&#x61&#x76&#x61&#x73&#x63&#x72&#x69&#x70&#x74&#x3A;alert("foo")\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        html = HTML(u'<IMG SRC=\'jav\tascript:alert("foo");\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())
        html = HTML(u'<IMG SRC=\'jav&#x09;ascript:alert("foo");\'>')
        self.assertEqual('<img/>', (html | HTMLSanitizer()).render())

    def test_sanitize_expression(self):
        html = HTML(u'<div style="top:expression(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_capital_expression(self):
        html = HTML(u'<div style="top:EXPRESSION(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_url_with_javascript(self):
        html = HTML(u'<div style="background-image:url(javascript:alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_capital_url_with_javascript(self):
        html = HTML(u'<div style="background-image:URL(javascript:alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_unicode_escapes(self):
        html = HTML(u'<div style="top:exp\\72 ess\\000069 on(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_backslash_without_hex(self):
        html = HTML(u'<div style="top:e\\xp\\ression(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        input_str = u'<div style="top:e\\\\xp\\\\ression(alert())">XSS</div>'
        html = HTML(input_str)
        self.assertEqual(input_str, six.text_type(html | StyleSanitizer()))

    def test_sanitize_unsafe_props(self):
        html = HTML(u'<div style="POSITION:RELATIVE">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="behavior:url(test.htc)">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="-ms-behavior:url(test.htc) url(#obj)">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="-o-link:\'javascript:alert(1)\';-o-link-source:current">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="-moz-binding:url(xss.xbl)">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_negative_margin(self):
        html = HTML(u'<div style="margin-top:-9999px">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="margin:0 -9999px">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_css_hack(self):
        html = HTML(u'<div style="*position:static">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="_margin:-10px">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_property_name(self):
        html = HTML(u'<div style="display:none;border-left-color:red;user_defined:1;-moz-user-selct:-moz-all">prop</div>')
        self.assertEqual('<div style="display:none; border-left-color:red">prop</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_unicode_expression(self):
        html = HTML(u'<div style="top:ｅｘｐｒｅｓｓｉｏｎ(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="top:ＥＸＰＲＥＳＳＩＯＮ(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))
        html = HTML(u'<div style="top:expʀessɪoɴ(alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))

    def test_sanitize_unicode_url(self):
        html = HTML(u'<div style="background-image:uʀʟ(javascript:alert())">XSS</div>')
        self.assertEqual('<div>XSS</div>', six.text_type(html | StyleSanitizer()))