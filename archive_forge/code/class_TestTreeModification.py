from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestTreeModification(SoupTest):

    def test_attribute_modification(self):
        soup = self.soup('<a id="1"></a>')
        soup.a['id'] = 2
        assert soup.decode() == self.document_for('<a id="2"></a>')
        del soup.a['id']
        assert soup.decode() == self.document_for('<a></a>')
        soup.a['id2'] = 'foo'
        assert soup.decode() == self.document_for('<a id2="foo"></a>')

    def test_new_tag_creation(self):
        builder = builder_registry.lookup('html')()
        soup = self.soup('<body></body>', builder=builder)
        a = Tag(soup, builder, 'a')
        ol = Tag(soup, builder, 'ol')
        a['href'] = 'http://foo.com/'
        soup.body.insert(0, a)
        soup.body.insert(1, ol)
        assert soup.body.encode() == b'<body><a href="http://foo.com/"></a><ol></ol></body>'

    def test_append_to_contents_moves_tag(self):
        doc = '<p id="1">Don\'t leave me <b>here</b>.</p>\n                <p id="2">Don\'t leave!</p>'
        soup = self.soup(doc)
        second_para = soup.find(id='2')
        bold = soup.b
        soup.find(id='2').append(soup.b)
        assert bold.parent == second_para
        assert soup.decode() == self.document_for('<p id="1">Don\'t leave me .</p>\n<p id="2">Don\'t leave!<b>here</b></p>')

    def test_replace_with_returns_thing_that_was_replaced(self):
        text = '<a></a><b><c></c></b>'
        soup = self.soup(text)
        a = soup.a
        new_a = a.replace_with(soup.c)
        assert a == new_a

    def test_unwrap_returns_thing_that_was_replaced(self):
        text = '<a><b></b><c></c></a>'
        soup = self.soup(text)
        a = soup.a
        new_a = a.unwrap()
        assert a == new_a

    def test_replace_with_and_unwrap_give_useful_exception_when_tag_has_no_parent(self):
        soup = self.soup('<a><b>Foo</b></a><c>Bar</c>')
        a = soup.a
        a.extract()
        assert None == a.parent
        with pytest.raises(ValueError):
            a.unwrap()
        with pytest.raises(ValueError):
            a.replace_with(soup.c)

    def test_replace_tag_with_itself(self):
        text = '<a><b></b><c>Foo<d></d></c></a><a><e></e></a>'
        soup = self.soup(text)
        c = soup.c
        soup.c.replace_with(c)
        assert soup.decode() == self.document_for(text)

    def test_replace_tag_with_its_parent_raises_exception(self):
        text = '<a><b></b></a>'
        soup = self.soup(text)
        with pytest.raises(ValueError):
            soup.b.replace_with(soup.a)

    def test_insert_tag_into_itself_raises_exception(self):
        text = '<a><b></b></a>'
        soup = self.soup(text)
        with pytest.raises(ValueError):
            soup.a.insert(0, soup.a)

    def test_insert_beautifulsoup_object_inserts_children(self):
        """Inserting one BeautifulSoup object into another actually inserts all
        of its children -- you'll never combine BeautifulSoup objects.
        """
        soup = self.soup("<p>And now, a word:</p><p>And we're back.</p>")
        text = '<p>p2</p><p>p3</p>'
        to_insert = self.soup(text)
        soup.insert(1, to_insert)
        for i in soup.descendants:
            assert not isinstance(i, BeautifulSoup)
        p1, p2, p3, p4 = list(soup.children)
        assert 'And now, a word:' == p1.string
        assert 'p2' == p2.string
        assert 'p3' == p3.string
        assert "And we're back." == p4.string

    def test_replace_with_maintains_next_element_throughout(self):
        soup = self.soup('<p><a>one</a><b>three</b></p>')
        a = soup.a
        b = a.contents[0]
        a.insert(1, 'two')
        left, right = a.contents
        left.replaceWith('')
        right.replaceWith('')
        assert 'three' == soup.b.string

    def test_replace_final_node(self):
        soup = self.soup('<b>Argh!</b>')
        soup.find(string='Argh!').replace_with('Hooray!')
        new_text = soup.find(string='Hooray!')
        b = soup.b
        assert new_text.previous_element == b
        assert new_text.parent == b
        assert new_text.previous_element.next_element == new_text
        assert new_text.next_element == None

    def test_consecutive_text_nodes(self):
        soup = self.soup('<a><b>Argh!</b><c></c></a>')
        soup.b.insert(1, 'Hooray!')
        assert soup.decode() == self.document_for('<a><b>Argh!Hooray!</b><c></c></a>')
        new_text = soup.find(string='Hooray!')
        assert new_text.previous_element == 'Argh!'
        assert new_text.previous_element.next_element == new_text
        assert new_text.previous_sibling == 'Argh!'
        assert new_text.previous_sibling.next_sibling == new_text
        assert new_text.next_sibling == None
        assert new_text.next_element == soup.c

    def test_insert_string(self):
        soup = self.soup('<a></a>')
        soup.a.insert(0, 'bar')
        soup.a.insert(0, 'foo')
        assert ['foo', 'bar'] == soup.a.contents
        assert soup.a.contents[0].next_element == 'bar'

    def test_insert_tag(self):
        builder = self.default_builder()
        soup = self.soup('<a><b>Find</b><c>lady!</c><d></d></a>', builder=builder)
        magic_tag = Tag(soup, builder, 'magictag')
        magic_tag.insert(0, 'the')
        soup.a.insert(1, magic_tag)
        assert soup.decode() == self.document_for('<a><b>Find</b><magictag>the</magictag><c>lady!</c><d></d></a>')
        b_tag = soup.b
        assert b_tag.next_sibling == magic_tag
        assert magic_tag.previous_sibling == b_tag
        find = b_tag.find(string='Find')
        assert find.next_element == magic_tag
        assert magic_tag.previous_element == find
        c_tag = soup.c
        assert magic_tag.next_sibling == c_tag
        assert c_tag.previous_sibling == magic_tag
        the = magic_tag.find(string='the')
        assert the.parent == magic_tag
        assert the.next_element == c_tag
        assert c_tag.previous_element == the

    def test_append_child_thats_already_at_the_end(self):
        data = '<a><b></b></a>'
        soup = self.soup(data)
        soup.a.append(soup.b)
        assert data == soup.decode()

    def test_extend(self):
        data = '<a><b><c><d><e><f><g></g></f></e></d></c></b></a>'
        soup = self.soup(data)
        l = [soup.g, soup.f, soup.e, soup.d, soup.c, soup.b]
        soup.a.extend(l)
        assert '<a><g></g><f></f><e></e><d></d><c></c><b></b></a>' == soup.decode()

    @pytest.mark.parametrize('get_tags', [lambda tag: tag, lambda tag: tag.contents])
    def test_extend_with_another_tags_contents(self, get_tags):
        data = '<body><div id="d1"><a>1</a><a>2</a><a>3</a><a>4</a></div><div id="d2"></div></body>'
        soup = self.soup(data)
        d1 = soup.find('div', id='d1')
        d2 = soup.find('div', id='d2')
        tags = get_tags(d1)
        d2.extend(tags)
        assert '<div id="d1"></div>' == d1.decode()
        assert '<div id="d2"><a>1</a><a>2</a><a>3</a><a>4</a></div>' == d2.decode()

    def test_move_tag_to_beginning_of_parent(self):
        data = '<a><b></b><c></c><d></d></a>'
        soup = self.soup(data)
        soup.a.insert(0, soup.d)
        assert '<a><d></d><b></b><c></c></a>' == soup.decode()

    def test_insert_works_on_empty_element_tag(self):
        soup = self.soup('<br/>')
        soup.br.insert(1, 'Contents')
        assert str(soup.br) == '<br>Contents</br>'

    def test_insert_before(self):
        soup = self.soup('<a>foo</a><b>bar</b>')
        soup.b.insert_before('BAZ')
        soup.a.insert_before('QUUX')
        assert soup.decode() == self.document_for('QUUX<a>foo</a>BAZ<b>bar</b>')
        soup.a.insert_before(soup.b)
        assert soup.decode() == self.document_for('QUUX<b>bar</b><a>foo</a>BAZ')
        b = soup.b
        with pytest.raises(ValueError):
            b.insert_before(b)
        b.extract()
        with pytest.raises(ValueError):
            b.insert_before('nope')
        soup = self.soup('<a>')
        soup.a.insert_before(soup.new_tag('a'))

    def test_insert_multiple_before(self):
        soup = self.soup('<a>foo</a><b>bar</b>')
        soup.b.insert_before('BAZ', ' ', 'QUUX')
        soup.a.insert_before('QUUX', ' ', 'BAZ')
        assert soup.decode() == self.document_for('QUUX BAZ<a>foo</a>BAZ QUUX<b>bar</b>')
        soup.a.insert_before(soup.b, 'FOO')
        assert soup.decode() == self.document_for('QUUX BAZ<b>bar</b>FOO<a>foo</a>BAZ QUUX')

    def test_insert_after(self):
        soup = self.soup('<a>foo</a><b>bar</b>')
        soup.b.insert_after('BAZ')
        soup.a.insert_after('QUUX')
        assert soup.decode() == self.document_for('<a>foo</a>QUUX<b>bar</b>BAZ')
        soup.b.insert_after(soup.a)
        assert soup.decode() == self.document_for('QUUX<b>bar</b><a>foo</a>BAZ')
        b = soup.b
        with pytest.raises(ValueError):
            b.insert_after(b)
        b.extract()
        with pytest.raises(ValueError):
            b.insert_after('nope')
        soup = self.soup('<a>')
        soup.a.insert_before(soup.new_tag('a'))

    def test_insert_multiple_after(self):
        soup = self.soup('<a>foo</a><b>bar</b>')
        soup.b.insert_after('BAZ', ' ', 'QUUX')
        soup.a.insert_after('QUUX', ' ', 'BAZ')
        assert soup.decode() == self.document_for('<a>foo</a>QUUX BAZ<b>bar</b>BAZ QUUX')
        soup.b.insert_after(soup.a, 'FOO ')
        assert soup.decode() == self.document_for('QUUX BAZ<b>bar</b><a>foo</a>FOO BAZ QUUX')

    def test_insert_after_raises_exception_if_after_has_no_meaning(self):
        soup = self.soup('')
        tag = soup.new_tag('a')
        string = soup.new_string('')
        with pytest.raises(ValueError):
            string.insert_after(tag)
        with pytest.raises(NotImplementedError):
            soup.insert_after(tag)
        with pytest.raises(ValueError):
            tag.insert_after(tag)

    def test_insert_before_raises_notimplementederror_if_before_has_no_meaning(self):
        soup = self.soup('')
        tag = soup.new_tag('a')
        string = soup.new_string('')
        with pytest.raises(ValueError):
            string.insert_before(tag)
        with pytest.raises(NotImplementedError):
            soup.insert_before(tag)
        with pytest.raises(ValueError):
            tag.insert_before(tag)

    def test_replace_with(self):
        soup = self.soup("<p>There's <b>no</b> business like <b>show</b> business</p>")
        no, show = soup.find_all('b')
        show.replace_with(no)
        assert soup.decode() == self.document_for("<p>There's  business like <b>no</b> business</p>")
        assert show.parent == None
        assert no.parent == soup.p
        assert no.next_element == 'no'
        assert no.next_sibling == ' business'

    def test_replace_with_errors(self):
        a_tag = Tag(name='a')
        with pytest.raises(ValueError):
            a_tag.replace_with("won't work")
        a_tag = self.soup('<a><b></b></a>').a
        with pytest.raises(ValueError):
            a_tag.b.replace_with(a_tag)
        with pytest.raises(ValueError):
            a_tag.b.replace_with('string1', a_tag, 'string2')

    def test_replace_with_multiple(self):
        data = '<a><b></b><c></c></a>'
        soup = self.soup(data)
        d_tag = soup.new_tag('d')
        d_tag.string = 'Text In D Tag'
        e_tag = soup.new_tag('e')
        f_tag = soup.new_tag('f')
        a_string = 'Random Text'
        soup.c.replace_with(d_tag, e_tag, a_string, f_tag)
        assert soup.decode() == '<a><b></b><d>Text In D Tag</d><e></e>Random Text<f></f></a>'
        assert soup.b.next_element == d_tag
        assert d_tag.string.next_element == e_tag
        assert e_tag.next_element.string == a_string
        assert e_tag.next_element.next_element == f_tag

    def test_replace_first_child(self):
        data = '<a><b></b><c></c></a>'
        soup = self.soup(data)
        soup.b.replace_with(soup.c)
        assert '<a><c></c></a>' == soup.decode()

    def test_replace_last_child(self):
        data = '<a><b></b><c></c></a>'
        soup = self.soup(data)
        soup.c.replace_with(soup.b)
        assert '<a><b></b></a>' == soup.decode()

    def test_nested_tag_replace_with(self):
        soup = self.soup('<a>We<b>reserve<c>the</c><d>right</d></b></a><e>to<f>refuse</f><g>service</g></e>')
        remove_tag = soup.b
        move_tag = soup.f
        remove_tag.replace_with(move_tag)
        assert soup.decode() == self.document_for('<a>We<f>refuse</f></a><e>to<g>service</g></e>')
        assert remove_tag.parent == None
        assert remove_tag.find(string='right').next_element == None
        assert remove_tag.previous_element == None
        assert remove_tag.next_sibling == None
        assert remove_tag.previous_sibling == None
        assert move_tag.parent == soup.a
        assert move_tag.previous_element == 'We'
        assert move_tag.next_element.next_element == soup.e
        assert move_tag.next_sibling == None
        to_text = soup.find(string='to')
        g_tag = soup.g
        assert to_text.next_element == g_tag
        assert to_text.next_sibling == g_tag
        assert g_tag.previous_element == to_text
        assert g_tag.previous_sibling == to_text

    def test_unwrap(self):
        tree = self.soup('\n            <p>Unneeded <em>formatting</em> is unneeded</p>\n            ')
        tree.em.unwrap()
        assert tree.em == None
        assert tree.p.text == 'Unneeded formatting is unneeded'

    def test_wrap(self):
        soup = self.soup('I wish I was bold.')
        value = soup.string.wrap(soup.new_tag('b'))
        assert value.decode() == '<b>I wish I was bold.</b>'
        assert soup.decode() == self.document_for('<b>I wish I was bold.</b>')

    def test_wrap_extracts_tag_from_elsewhere(self):
        soup = self.soup('<b></b>I wish I was bold.')
        soup.b.next_sibling.wrap(soup.b)
        assert soup.decode() == self.document_for('<b>I wish I was bold.</b>')

    def test_wrap_puts_new_contents_at_the_end(self):
        soup = self.soup('<b>I like being bold.</b>I wish I was bold.')
        soup.b.next_sibling.wrap(soup.b)
        assert 2 == len(soup.b.contents)
        assert soup.decode() == self.document_for('<b>I like being bold.I wish I was bold.</b>')

    def test_extract(self):
        soup = self.soup('<html><body>Some content. <div id="nav">Nav crap</div> More content.</body></html>')
        assert len(soup.body.contents) == 3
        extracted = soup.find(id='nav').extract()
        assert soup.decode() == '<html><body>Some content.  More content.</body></html>'
        assert extracted.decode() == '<div id="nav">Nav crap</div>'
        assert len(soup.body.contents) == 2
        assert extracted.parent == None
        assert extracted.previous_element == None
        assert extracted.next_element.next_element == None
        content_1 = soup.find(string='Some content. ')
        content_2 = soup.find(string=' More content.')
        assert content_1.next_element == content_2
        assert content_1.next_sibling == content_2
        assert content_2.previous_element == content_1
        assert content_2.previous_sibling == content_1

    def test_extract_distinguishes_between_identical_strings(self):
        soup = self.soup('<a>foo</a><b>bar</b>')
        foo_1 = soup.a.string
        bar_1 = soup.b.string
        foo_2 = soup.new_string('foo')
        bar_2 = soup.new_string('bar')
        soup.a.append(foo_2)
        soup.b.append(bar_2)
        foo_1.extract()
        bar_2.extract()
        assert foo_2 == soup.a.string
        assert bar_2 == soup.b.string

    def test_extract_multiples_of_same_tag(self):
        soup = self.soup('\n<html>\n<head>\n<script>foo</script>\n</head>\n<body>\n <script>bar</script>\n <a></a>\n</body>\n<script>baz</script>\n</html>')
        [soup.script.extract() for i in soup.find_all('script')]
        assert '<body>\n\n<a></a>\n</body>' == str(soup.body)

    def test_extract_works_when_element_is_surrounded_by_identical_strings(self):
        soup = self.soup('<html>\n<body>hi</body>\n</html>')
        soup.find('body').extract()
        assert None == soup.find('body')

    def test_clear(self):
        """Tag.clear()"""
        soup = self.soup('<p><a>String <em>Italicized</em></a> and another</p>')
        a = soup.a
        soup.p.clear()
        assert len(soup.p.contents) == 0
        assert hasattr(a, 'contents')
        em = a.em
        a.clear(decompose=True)
        assert 0 == len(em.contents)

    def test_decompose(self):
        soup = self.soup('<p><a>String <em>Italicized</em></a></p><p>Another para</p>')
        p1, p2 = soup.find_all('p')
        a = p1.a
        text = p1.em.string
        for i in [p1, p2, a, text]:
            assert False == i.decomposed
        p1.decompose()
        for i in [p1, a, text]:
            assert True == i.decomposed
        assert False == p2.decomposed

    def test_string_set(self):
        """Tag.string = 'string'"""
        soup = self.soup('<a></a> <b><c></c></b>')
        soup.a.string = 'foo'
        assert soup.a.contents == ['foo']
        soup.b.string = 'bar'
        assert soup.b.contents == ['bar']

    def test_string_set_does_not_affect_original_string(self):
        soup = self.soup('<a><b>foo</b><c>bar</c>')
        soup.b.string = soup.c.string
        assert soup.a.encode() == b'<a><b>bar</b><c>bar</c></a>'

    def test_set_string_preserves_class_of_string(self):
        soup = self.soup('<a></a>')
        cdata = CData('foo')
        soup.a.string = cdata
        assert isinstance(soup.a.string, CData)