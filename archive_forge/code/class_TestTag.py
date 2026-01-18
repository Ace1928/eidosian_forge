import warnings
from bs4.element import (
from . import SoupTest
class TestTag(SoupTest):
    """Test various methods of Tag which aren't so complicated they
    need their own classes.
    """

    def test__should_pretty_print(self):
        tag = self.soup('').new_tag('a_tag')
        tag._preserve_whitespace_tags = None
        assert True == tag._should_pretty_print(0)
        tag.preserve_whitespace_tags = ['some_other_tag']
        assert True == tag._should_pretty_print(1)
        assert False == tag._should_pretty_print(None)
        tag.preserve_whitespace_tags = ['some_other_tag', 'a_tag']
        assert False == tag._should_pretty_print(1)

    def test_len(self):
        """The length of a Tag is its number of children."""
        soup = self.soup('<top>1<b>2</b>3</top>')
        assert len(soup.contents) == 1
        assert len(soup) == 1
        assert len(soup.top) == 3
        assert len(soup.top.contents) == 3

    def test_member_access_invokes_find(self):
        """Accessing a Python member .foo invokes find('foo')"""
        soup = self.soup('<b><i></i></b>')
        assert soup.b == soup.find('b')
        assert soup.b.i == soup.find('b').find('i')
        assert soup.a == None

    def test_deprecated_member_access(self):
        soup = self.soup('<b><i></i></b>')
        with warnings.catch_warnings(record=True) as w:
            tag = soup.bTag
        assert soup.b == tag
        assert '.bTag is deprecated, use .find("b") instead. If you really were looking for a tag called bTag, use .find("bTag")' == str(w[0].message)

    def test_has_attr(self):
        """has_attr() checks for the presence of an attribute.

        Please note note: has_attr() is different from
        __in__. has_attr() checks the tag's attributes and __in__
        checks the tag's chidlren.
        """
        soup = self.soup("<foo attr='bar'>")
        assert soup.foo.has_attr('attr')
        assert not soup.foo.has_attr('attr2')

    def test_attributes_come_out_in_alphabetical_order(self):
        markup = '<b a="1" z="5" m="3" f="2" y="4"></b>'
        self.assertSoupEquals(markup, '<b a="1" f="2" m="3" y="4" z="5"></b>')

    def test_string(self):
        soup = self.soup('<b>foo</b>')
        assert soup.b.string == 'foo'

    def test_empty_tag_has_no_string(self):
        soup = self.soup('<b></b>')
        assert soup.b.string == None

    def test_tag_with_multiple_children_has_no_string(self):
        soup = self.soup('<a>foo<b></b><b></b></b>')
        assert soup.b.string == None
        soup = self.soup('<a>foo<b></b>bar</b>')
        assert soup.b.string == None
        soup = self.soup('<a>foo</b>')
        soup.a.insert(1, 'bar')
        assert soup.a.string == None

    def test_tag_with_recursive_string_has_string(self):
        soup = self.soup('<a><b>foo</b></a>')
        assert soup.a.string == 'foo'
        assert soup.string == 'foo'

    def test_lack_of_string(self):
        """Only a Tag containing a single text node has a .string."""
        soup = self.soup('<b>f<i>e</i>o</b>')
        assert soup.b.string is None
        soup = self.soup('<b></b>')
        assert soup.b.string is None

    def test_all_text(self):
        """Tag.text and Tag.get_text(sep=u"") -> all child text, concatenated"""
        soup = self.soup('<a>a<b>r</b>   <r> t </r></a>')
        assert soup.a.text == 'ar  t '
        assert soup.a.get_text(strip=True) == 'art'
        assert soup.a.get_text(',') == 'a,r, , t '
        assert soup.a.get_text(',', strip=True) == 'a,r,t'

    def test_get_text_ignores_special_string_containers(self):
        soup = self.soup('foo<!--IGNORE-->bar')
        assert soup.get_text() == 'foobar'
        assert soup.get_text(types=(NavigableString, Comment)) == 'fooIGNOREbar'
        assert soup.get_text(types=None) == 'fooIGNOREbar'
        soup = self.soup('foo<style>CSS</style><script>Javascript</script>bar')
        assert soup.get_text() == 'foobar'

    def test_all_strings_ignores_special_string_containers(self):
        soup = self.soup('foo<!--IGNORE-->bar')
        assert ['foo', 'bar'] == list(soup.strings)
        soup = self.soup('foo<style>CSS</style><script>Javascript</script>bar')
        assert ['foo', 'bar'] == list(soup.strings)

    def test_string_methods_inside_special_string_container_tags(self):
        style = self.soup('<div>a<style>Some CSS</style></div>')
        template = self.soup('<div>a<template><p>Templated <b>text</b>.</p><!--With a comment.--></template></div>')
        script = self.soup('<div>a<script><!--a comment-->Some text</script></div>')
        assert style.div.get_text() == 'a'
        assert list(style.div.strings) == ['a']
        assert style.div.style.get_text() == 'Some CSS'
        assert list(style.div.style.strings) == ['Some CSS']
        assert template.div.get_text() == 'a'
        assert list(template.div.strings) == ['a']
        assert template.div.template.get_text() == 'Templated text.'
        assert list(template.div.template.strings) == ['Templated ', 'text', '.']
        assert script.div.get_text() == 'a'
        assert list(script.div.strings) == ['a']
        assert script.div.script.get_text() == '<!--a comment-->Some text'
        assert list(script.div.script.strings) == ['<!--a comment-->Some text']