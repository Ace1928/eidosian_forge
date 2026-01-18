from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
class TestProtected(TextTestMixin, TestCase):

    def test__init__(self):
        assert six.text_type(Protected('a', '', 'c')) == 'ac'
        assert six.text_type(Protected('a', Text(), 'c')) == 'ac'
        text = Protected(Protected(), Protected('mary ', 'had ', 'a little lamb'))
        assert text == Protected(Protected('mary had a little lamb'))
        assert six.text_type(text) == 'mary had a little lamb'
        text = six.text_type(Protected('a', Protected('b', 'c'), Tag('em', 'x'), Symbol('nbsp'), 'd'))
        assert text == 'abcx<nbsp>d'
        with pytest.raises(ValueError):
            Protected({})
        with pytest.raises(ValueError):
            Protected(0, 0)

    def test__eq__(self):
        assert Protected() == Protected()
        assert not Protected() != Protected()
        assert Protected('Cat') == Protected('Cat')
        assert not Protected('Cat') != Protected('Cat')
        assert Protected('Cat', ' tail') == Protected('Cat tail')
        assert not Protected('Cat', ' tail') != Protected('Cat tail')
        assert Protected('Cat') != Protected('Dog')
        assert not Protected('Cat') == Protected('Dog')

    def test__len__(self):
        assert len(Protected()) == 0
        assert len(Protected('Never', ' ', 'Knows', ' ', 'Best')) == len('Never Knows Best')
        assert len(Protected('Never', ' ', Tag('em', 'Knows', ' '), 'Best')) == len('Never Knows Best')
        assert len(Protected('Never', ' ', Tag('em', HRef('/', 'Knows'), ' '), 'Best')) == len('Never Knows Best')

    def test__str__(self):
        assert six.text_type(Protected()) == ''
        assert six.text_type(Protected(u'Чудаки украшают мир')) == u'Чудаки украшают мир'

    def test__contains__(self):
        text = Protected('mary ', 'had ', 'a little lamb')
        assert 'mary' in text
        assert 'Mary' not in text
        assert 'had a little' in text
        text = Protected('a', 'b', 'c')
        assert 'abc' in text

    def test_capfirst(self):
        text = Protected('mary ', 'had ', 'a Little Lamb')
        assert six.text_type(text.capitalize()) == 'mary had a Little Lamb'

    def test_capitalize(self):
        text = Protected('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.capitalize()) == 'mary had a little lamb'

    def test__add__(self):
        t = Protected('a')
        assert t + 'b' == Text(Protected('a'), 'b')
        assert t + t == Text(Protected('aa'))

    def test__getitem__(self):
        t = Protected('1234567890')
        with pytest.raises(TypeError):
            1 in t
        assert t == Protected('1234567890')
        assert t[:0] == Protected('')
        assert t[:1] == Protected('1')
        assert t[:3] == Protected('123')
        assert t[:5] == Protected('12345')
        assert t[:7] == Protected('1234567')
        assert t[:10] == Protected('1234567890')
        assert t[:100] == Protected('1234567890')
        assert t[:-100] == Protected('')
        assert t[:-10] == Protected('')
        assert t[:-9] == Protected('1')
        assert t[:-7] == Protected('123')
        assert t[:-5] == Protected('12345')
        assert t[:-3] == Protected('1234567')
        assert t[-100:] == Protected('1234567890')
        assert t[-10:] == Protected('1234567890')
        assert t[-9:] == Protected('234567890')
        assert t[-7:] == Protected('4567890')
        assert t[-5:] == Protected('67890')
        assert t[-3:] == Protected('890')
        assert t[1:] == Protected('234567890')
        assert t[3:] == Protected('4567890')
        assert t[5:] == Protected('67890')
        assert t[7:] == Protected('890')
        assert t[10:] == Protected('')
        assert t[100:] == Protected('')
        assert t[0:10] == Protected('1234567890')
        assert t[0:100] == Protected('1234567890')
        assert t[2:3] == Protected('3')
        assert t[2:4] == Protected('34')
        assert t[3:7] == Protected('4567')
        assert t[4:7] == Protected('567')
        assert t[4:7] == Protected('567')
        assert t[7:9] == Protected('89')
        assert t[100:200] == Protected('')
        t = Protected('123', Protected('456', Protected('789')), '0')
        assert t[:3] == Protected('123')
        assert t[:5] == Protected('123', Protected('45'))
        assert t[:7] == Protected('123', Protected('456', Protected('7')))
        assert t[:10] == Protected('123', Protected('456', Protected('789')), '0')
        assert t[:100] == Protected('123', Protected('456', Protected('789')), '0')
        assert t[:-7] == Protected('123')
        assert t[:-5] == Protected('123', Protected('45'))
        assert t[:-3] == Protected('123', Protected('456', Protected('7')))

    def test_append(self):
        text = Protected('Chuck Norris')
        assert (text + ' wins!').render_as('latex') == '{Chuck Norris} wins!'
        assert text.append(' wins!').render_as('latex') == '{Chuck Norris wins!}'

    def test_upper(self):
        text = Protected('Mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'Mary had a little lamb'
        text = Protected('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'mary had a little lamb'

    def test_lower(self):
        text = Protected('Mary ', 'had ', 'a little lamb')
        assert six.text_type(text.lower()) == 'Mary had a little lamb'
        text = Protected('MARY ', 'HAD ', 'A LITTLE LAMB')
        assert six.text_type(text.lower()) == 'MARY HAD A LITTLE LAMB'

    def test_startswith(self):
        assert not Protected().startswith('.')
        assert not Protected().startswith(('.', '!'))
        text = Protected('mary ', 'had ', 'a little lamb')
        assert not text.startswith('M')
        assert text.startswith('m')
        text = Protected('a', 'b', 'c')
        assert text.startswith('ab')
        assert Protected('This is good').startswith(('This', 'That'))
        assert not Protected('This is good').startswith(('That', 'Those'))

    def test_endswith(self):
        assert not Protected().endswith('.')
        assert not Protected().endswith(('.', '!'))
        text = Protected('mary ', 'had ', 'a little lamb')
        assert not text.endswith('B')
        assert text.endswith('b')
        text = Protected('a', 'b', 'c')
        assert text.endswith('bc')
        assert Protected('This is good').endswith(('good', 'wonderful'))
        assert not Protected('This is good').endswith(('bad', 'awful'))

    def test_isalpha(self):
        assert not Protected().isalpha()
        assert not Protected('a b c').isalpha()
        assert Protected('abc').isalpha()
        assert Protected(u'文字').isalpha()

    def test_join(self):
        assert Protected(' ').join(['a', Protected('b c')]).render_as('latex') == 'a{ b c}'
        assert Protected(nbsp).join(['a', 'b', 'c']).render_as('latex') == 'a{~}b{~}c'
        assert nbsp.join(['a', Protected('b'), 'c']).render_as('latex') == 'a~{b}~c'
        assert String('-').join([Protected('a'), Protected('b'), Protected('c')]).render_as('latex') == '{a}-{b}-{c}'
        result = Protected(' and ').join(['a', 'b', 'c']).render_as('latex')
        assert result == 'a{ and }b{ and }c'

    def test_split(self):
        assert Protected().split() == [Protected()]
        assert Protected().split('abc') == [Protected()]
        assert Protected('a').split() == [Protected('a')]
        assert Protected('a ').split() == [Protected('a ')]
        assert Protected('   a   ').split() == [Protected('   a   ')]
        assert Protected('a + b').split() == [Protected('a + b')]
        assert Protected('a + b').split(' + ') == [Protected('a + b')]
        assert Protected('abc').split('xyz') == [Protected('abc')]
        assert Protected('---').split('--') == [Protected('---')]
        assert Protected('---').split('-') == [Protected('---')]

    def test_add_period(self):
        assert not Protected().endswith(('.', '!', '?'))
        assert not textutils.is_terminated(Protected())
        assert Protected().add_period().render_as('latex') == '{}'
        text = Protected("That's all, folks")
        assert text.add_period().render_as('latex') == "{That's all, folks.}"

    def test_render_as(self):
        string = Protected('a < b')
        assert string.render_as('latex') == '{a < b}'
        assert string.render_as('html') == '<span class="bibtex-protected">a &lt; b</span>'