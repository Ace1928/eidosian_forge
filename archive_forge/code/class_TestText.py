from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
class TestText(TextTestMixin, TestCase):

    def test__init__(self):
        assert six.text_type(Text('a', '', 'c')) == 'ac'
        assert six.text_type(Text('a', Text(), 'c')) == 'ac'
        text = Text(Text(), Text('mary ', 'had ', 'a little lamb'))
        assert six.text_type(text) == 'mary had a little lamb'
        text = six.text_type(Text('a', Text('b', 'c'), Tag('em', 'x'), Symbol('nbsp'), 'd'))
        assert text == 'abcx<nbsp>d'
        with pytest.raises(ValueError):
            Text({})
        with pytest.raises(ValueError):
            Text(0, 0)

    def test__eq__(self):
        assert Text() == Text()
        assert not Text() != Text()
        assert Text('Cat') == Text('Cat')
        assert not Text('Cat') != Text('Cat')
        assert Text('Cat', ' tail') == Text('Cat tail')
        assert not Text('Cat', ' tail') != Text('Cat tail')
        assert Text('Cat') != Text('Dog')
        assert not Text('Cat') == Text('Dog')

    def test__len__(self):
        assert len(Text()) == 0
        assert len(Text('Never', ' ', 'Knows', ' ', 'Best')) == len('Never Knows Best')
        assert len(Text('Never', ' ', Tag('em', 'Knows', ' '), 'Best')) == len('Never Knows Best')
        assert len(Text('Never', ' ', Tag('em', HRef('/', 'Knows'), ' '), 'Best')) == len('Never Knows Best')

    def test__str__(self):
        assert six.text_type(Text()) == ''
        assert six.text_type(Text(u'Чудаки украшают мир')) == u'Чудаки украшают мир'

    def test__contains__(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert 'mary' in text
        assert not 'Mary' in text
        assert 'had a little' in text
        text = Text('a', 'b', 'c')
        assert 'abc' in text

    def test_capfirst(self):
        text = Text('dear ', 'Alice')
        assert six.text_type(text.capfirst()) == 'Dear Alice'

    def test_capitalize(self):
        text = Text('mary ', 'had ', 'a Little Lamb')
        assert six.text_type(text.capitalize()) == 'Mary had a little lamb'

    def test__add__(self):
        t = Text('a')
        assert six.text_type(t + 'b') == 'ab'
        assert six.text_type(t + t) == 'aa'
        assert six.text_type(t) == 'a'

    def test__getitem__(self):
        t = Text('123', Text('456', Text('78'), '9'), '0')
        with pytest.raises(TypeError):
            1 in t
        assert t == Text('1234567890')
        assert t[:0] == Text('')
        assert t[:1] == Text('1')
        assert t[:3] == Text('123')
        assert t[:5] == Text('12345')
        assert t[:7] == Text('1234567')
        assert t[:10] == Text('1234567890')
        assert t[:100] == Text('1234567890')
        assert t[:-100] == Text('')
        assert t[:-10] == Text('')
        assert t[:-9] == Text('1')
        assert t[:-7] == Text('123')
        assert t[:-5] == Text('12345')
        assert t[:-3] == Text('1234567')
        assert t[-100:] == Text('1234567890')
        assert t[-10:] == Text('1234567890')
        assert t[-9:] == Text('234567890')
        assert t[-7:] == Text('4567890')
        assert t[-5:] == Text('67890')
        assert t[-3:] == Text('890')
        assert t[1:] == Text('234567890')
        assert t[3:] == Text('4567890')
        assert t[5:] == Text('67890')
        assert t[7:] == Text('890')
        assert t[10:] == Text('')
        assert t[100:] == Text('')
        assert t[0:10] == Text('1234567890')
        assert t[0:100] == Text('1234567890')
        assert t[2:3] == Text('3')
        assert t[2:4] == Text('34')
        assert t[3:7] == Text('4567')
        assert t[4:7] == Text('567')
        assert t[4:7] == Text('567')
        assert t[7:9] == Text('89')
        assert t[100:200] == Text('')
        t = Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:3] == Text('123')
        assert t[:5] == Text('123', Tag('em', '45'))
        assert t[:7] == Text('123', Tag('em', '456', HRef('/', '7')))
        assert t[:10] == Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:100] == Text('123', Tag('em', '456', HRef('/', '789')), '0')
        assert t[:-7] == Text('123')
        assert t[:-5] == Text('123', Tag('em', '45'))
        assert t[:-3] == Text('123', Tag('em', '456', HRef('/', '7')))

    def test_append(self):
        text = Tag('strong', 'Chuck Norris')
        assert (text + ' wins!').render_as('html') == '<strong>Chuck Norris</strong> wins!'
        assert text.append(' wins!').render_as('html') == '<strong>Chuck Norris wins!</strong>'
        text = HRef('/', 'Chuck Norris')
        assert (text + ' wins!').render_as('html') == '<a href="/">Chuck Norris</a> wins!'
        assert text.append(' wins!').render_as('html') == '<a href="/">Chuck Norris wins!</a>'

    def test_upper(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.upper()) == 'MARY HAD A LITTLE LAMB'

    def test_lower(self):
        text = Text('mary ', 'had ', 'a little lamb')
        assert six.text_type(text.lower()) == 'mary had a little lamb'

    def test_startswith(self):
        assert not Text().startswith('.')
        assert not Text().startswith(('.', '!'))
        text = Text('mary ', 'had ', 'a little lamb')
        assert not text.startswith('M')
        assert text.startswith('m')
        text = Text('a', 'b', 'c')
        assert text.startswith('ab')
        assert Text('This is good').startswith(('This', 'That'))
        assert not Text('This is good').startswith(('That', 'Those'))

    def test_endswith(self):
        assert not Text().endswith('.')
        assert not Text().endswith(('.', '!'))
        text = Text('mary ', 'had ', 'a little lamb')
        assert not text.endswith('B')
        assert text.endswith('b')
        text = Text('a', 'b', 'c')
        assert text.endswith('bc')
        assert Text('This is good').endswith(('good', 'wonderful'))
        assert not Text('This is good').endswith(('bad', 'awful'))

    def test_isalpha(self):
        assert not Text().isalpha()
        assert not Text('a b c').isalpha()
        assert Text('abc').isalpha()
        assert Text(u'文字').isalpha()
        assert Text('ab', Tag('em', 'cd'), 'ef').isalpha()
        assert not Text('ab', Tag('em', '12'), 'ef').isalpha()

    def test_join(self):
        assert six.text_type(Text(' ').join(['a', Text('b c')])) == 'a b c'
        assert six.text_type(Text(nbsp).join(['a', 'b', 'c'])) == 'a<nbsp>b<nbsp>c'
        assert six.text_type(nbsp.join(['a', 'b', 'c'])) == 'a<nbsp>b<nbsp>c'
        assert six.text_type(String('-').join(['a', 'b', 'c'])) == 'a-b-c'
        result = Tag('em', ' and ').join(['a', 'b', 'c']).render_as('html')
        assert result == 'a<em> and </em>b<em> and </em>c'
        result = HRef('/', ' and ').join(['a', 'b', 'c']).render_as('html')
        assert result == 'a<a href="/"> and </a>b<a href="/"> and </a>c'

    def test_split(self):
        assert Text().split() == []
        assert Text().split('abc') == [Text()]
        assert Text('a').split() == [Text('a')]
        assert Text('a ').split() == [Text('a')]
        assert Text('   a   ').split() == [Text('a')]
        assert Text('a + b').split() == [Text('a'), Text('+'), Text('b')]
        assert Text('a + b').split(' + ') == [Text('a'), Text('b')]
        assert Text('a + b').split(re.compile('\\s')) == [Text('a'), Text('+'), Text('b')]
        assert Text('abc').split('xyz') == [Text('abc')]
        assert Text('---').split('--') == [Text(), Text('-')]
        assert Text('---').split('-') == [Text(), Text(), Text(), Text()]

    def test_add_period(self):
        assert Text().endswith(('.', '!', '?')) == False
        assert textutils.is_terminated(Text()) == False
        assert six.text_type(Text().add_period()) == ''
        text = Text("That's all, folks")
        assert six.text_type(text.add_period()) == "That's all, folks."

    def test_render_as(self):
        string = Text(u'Detektivbyrån & friends')
        assert string.render_as('text') == u'Detektivbyrån & friends'
        assert string.render_as('html') == u'Detektivbyrån &amp; friends'