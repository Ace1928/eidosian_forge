from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
class TestHRef(TextTestMixin, TestCase):

    def test__init__(self):
        empty = HRef('/')
        assert empty.url == '/'
        assert empty.parts == []

    def test__str__(self):
        empty = HRef('/')
        assert six.text_type(empty) == ''
        text = Text('This ', HRef('/', 'is'), ' good')
        six.text_type(text) == 'This is good'

    def test__eq__(self):
        assert HRef('/', '') != ''
        assert HRef('/', '') != Text()
        assert HRef('/', '') != HRef('', '')
        assert HRef('/', '') == HRef('/', '')
        assert HRef('/', 'good') != HRef('', 'bad')
        assert HRef('/', 'good') != Text('good')
        assert HRef('/', 'good') == HRef('/', 'good')
        assert not HRef('/', 'good') != HRef('/', 'good')
        assert HRef('strong', '') != Tag('strong', '')

    def test__len__(self):
        val = 'Tomato apple!'
        assert len(HRef('index', val)) == len(val)

    def test__contains__(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert not 'mary' in tag
        assert 'Mary' in tag
        assert 'had a little' in tag
        text = Text('This ', HRef('/', 'is'), ' good')
        assert not 'This is' in text

    def test__getitem__(self):
        t = HRef('/', '1234567890')
        with pytest.raises(TypeError):
            1 in t
        assert t == HRef('/', '1234567890')
        assert t[:] == t
        assert t[:0] == HRef('/', '')
        assert t[:1] == HRef('/', '1')
        assert t[:3] == HRef('/', '123')
        assert t[:5] == HRef('/', '12345')
        assert t[:7] == HRef('/', '1234567')
        assert t[:10] == HRef('/', '1234567890')
        assert t[:100] == HRef('/', '1234567890')
        assert t[:-100] == HRef('/', '')
        assert t[:-10] == HRef('/', '')
        assert t[:-9] == HRef('/', '1')
        assert t[:-7] == HRef('/', '123')
        assert t[:-5] == HRef('/', '12345')
        assert t[:-3] == HRef('/', '1234567')
        assert t[-100:] == HRef('/', '1234567890')
        assert t[-10:] == HRef('/', '1234567890')
        assert t[-9:] == HRef('/', '234567890')
        assert t[-7:] == HRef('/', '4567890')
        assert t[-5:] == HRef('/', '67890')
        assert t[-3:] == HRef('/', '890')
        assert t[1:] == HRef('/', '234567890')
        assert t[3:] == HRef('/', '4567890')
        assert t[5:] == HRef('/', '67890')
        assert t[7:] == HRef('/', '890')
        assert t[10:] == HRef('/', '')
        assert t[100:] == HRef('/', '')
        assert t[0:10] == HRef('/', '1234567890')
        assert t[0:100] == HRef('/', '1234567890')
        assert t[2:3] == HRef('/', '3')
        assert t[2:4] == HRef('/', '34')
        assert t[3:7] == HRef('/', '4567')
        assert t[4:7] == HRef('/', '567')
        assert t[4:7] == HRef('/', '567')
        assert t[7:9] == HRef('/', '89')
        assert t[100:200] == HRef('/', '')
        t = HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:3] == HRef('', '123')
        assert t[:5] == HRef('', '123', HRef('/', '45'))
        assert t[:7] == HRef('', '123', HRef('/', '456', HRef('/', '7')))
        assert t[:10] == HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:100] == HRef('', '123', HRef('/', '456', HRef('/', '789')), '0')
        assert t[:-7] == HRef('', '123')
        assert t[:-5] == HRef('', '123', HRef('/', '45'))
        assert t[:-3] == HRef('', '123', HRef('/', '456', HRef('/', '7')))

    def test__add__(self):
        assert HRef('/', '') + HRef('/', '') == Text(HRef('/', ''))
        assert HRef('/', '') + HRef('strong', '') == Text(HRef('/', ''), HRef('strong', ''))
        assert HRef('/', 'Good') + HRef('/', '') == Text(HRef('/', 'Good'))
        assert HRef('/', 'Good') + HRef('/', ' job!') == Text(HRef('/', 'Good job!'))
        assert HRef('/', 'Good') + HRef('strong', ' job!') == Text(HRef('/', 'Good'), HRef('strong', ' job!'))
        assert HRef('/', 'Good') + Text(' job!') == Text(HRef('/', 'Good'), ' job!')
        assert Text('Good') + HRef('/', ' job!') == Text('Good', HRef('/', ' job!'))

    def test_append(self):
        text = HRef('/', 'Chuck Norris')
        assert (text + ' wins!').render_as('html') == '<a href="/">Chuck Norris</a> wins!'
        assert text.append(' wins!').render_as('html') == '<a href="/">Chuck Norris wins!</a>'

    def test_lower(self):
        assert HRef('/').lower() == HRef('/')
        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.lower().render_as('latex') == '\\href{http://www.example.com}{hyperlinked text.}'
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.lower().render_as('html') == '<a href="info.html">mary had a little lamb</a>'

    def test_upper(self):
        assert HRef('/').upper() == HRef('/')
        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.upper().render_as('latex') == '\\href{http://www.example.com}{HYPERLINKED TEXT.}'
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.upper().render_as('html') == '<a href="info.html">MARY HAD A LITTLE LAMB</a>'

    def test_capfirst(self):
        assert HRef('/').capfirst() == Text(HRef('/'))
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a Little Lamb'))
        assert tag.capfirst().render_as('html') == '<a href="info.html">Mary had a Little Lamb</a>'
        assert tag.lower().capfirst().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'

    def test_capitalize(self):
        assert HRef('/').capitalize() == Text(HRef('/'))
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a Little Lamb'))
        assert tag.capitalize().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'
        assert tag.lower().capitalize().render_as('html') == '<a href="info.html">Mary had a little lamb</a>'

    def test_add_period(self):
        assert HRef('/').add_period() == HRef('/')
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.add_period().render_as('html') == '<a href="info.html">Mary had a little lamb.</a>'
        assert tag.add_period().add_period().render_as('html') == '<a href="info.html">Mary had a little lamb.</a>'

    def test_split(self):
        empty = HRef('/')
        assert empty.split() == []
        assert empty.split('abc') == [empty]
        href = HRef('/', 'World Wide Web')
        assert href.split() == [HRef('/', 'World'), HRef('/', 'Wide'), HRef('/', 'Web')]
        result = Text('Estimated size of the ', href).split()
        assert result == [Text('Estimated'), Text('size'), Text('of'), Text('the'), Text(HRef('/', 'World')), Text(HRef('/', 'Wide')), Text(HRef('/', 'Web'))]
        text = Text(Tag('em', Text(Tag('strong', HRef('/', '  Very, very'), ' bad'), ' guys')), '! ')
        assert text.render_as('html') == '<em><strong><a href="/">  Very, very</a> bad</strong> guys</em>! '
        assert text.split(', ') == [Text(Tag('em', Tag('strong', HRef('/', '  Very')))), Text(Tag('em', Tag('strong', HRef('/', 'very'), ' bad'), ' guys'), '! ')]
        assert text.split(' ') == [Text(), Text(), Text(Tag('em', Tag('strong', HRef('/', 'Very,')))), Text(Tag('em', Tag('strong', HRef('/', 'very')))), Text(Tag('em', Tag('strong', 'bad'))), Text(Tag('em', 'guys'), '!'), Text()]
        assert text.split(' ', keep_empty_parts=False) == [Text(Tag('em', Tag('strong', HRef('/', 'Very,')))), Text(Tag('em', Tag('strong', HRef('/', 'very')))), Text(Tag('em', Tag('strong', 'bad'))), Text(Tag('em', 'guys'), '!')]
        assert text.split() == [Text(Tag('em', Tag('strong', HRef('/', 'Very,')))), Text(Tag('em', Tag('strong', HRef('/', 'very')))), Text(Tag('em', Tag('strong', 'bad'))), Text(Tag('em', 'guys'), '!')]
        assert text.split(keep_empty_parts=True) == [Text(), Text(Tag('em', Tag('strong', HRef('/', 'Very,')))), Text(Tag('em', Tag('strong', HRef('/', 'very')))), Text(Tag('em', Tag('strong', 'bad'))), Text(Tag('em', 'guys'), '!'), Text()]
        text = Text(' A', Tag('em', ' big', HRef('/', ' ', Tag('strong', 'no-no'), '!  ')))
        assert text.render_as('html') == ' A<em> big<a href="/"> <strong>no-no</strong>!  </a></em>'
        assert text.split('-') == [Text(' A', Tag('em', ' big', HRef('/', ' ', Tag('strong', 'no')))), Text(Tag('em', HRef('/', Tag('strong', 'no'), '!  ')))]
        assert text.split(' ') == [Text(), Text('A'), Text(Tag('em', 'big')), Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))), Text(), Text()]
        assert text.split(' ', keep_empty_parts=False) == [Text('A'), Text(Tag('em', 'big')), Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!')))]
        assert text.split() == [Text('A'), Text(Tag('em', 'big')), Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!')))]
        assert text.split(keep_empty_parts=True) == [Text(), Text('A'), Text(Tag('em', 'big')), Text(Tag('em', HRef('/', Tag('strong', 'no-no'), '!'))), Text()]

    def test_join(self):
        href = HRef('/', 'World Wide Web')
        result = Text('-').join(Text('Estimated size of the ', href).split())
        assert result == Text('Estimated-size-of-the-', HRef('/', 'World'), '-', HRef('/', 'Wide'), '-', HRef('/', 'Web'))

    def test_startswith(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.startswith('')
        assert tag.startswith('M')
        assert tag.startswith('Mary')
        assert not tag.startswith('m')
        assert not tag.startswith('mary')
        tag = HRef('/', 'a', 'b', 'c')
        assert tag.startswith('ab')
        tag = HRef('/', 'This is good')
        assert tag.startswith(('This', 'That'))
        assert not tag.startswith(('That', 'Those'))
        text = Text('This ', HRef('/', 'is'), ' good')
        assert not text.startswith('This is')

    def test_endswith(self):
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.endswith('')
        assert not tag.endswith('B')
        assert tag.endswith('b')
        assert tag.endswith('lamb')
        tag = HRef('/', 'a', 'b', 'c')
        assert tag.endswith('bc')
        tag = HRef('/', 'This is good')
        assert tag.endswith(('good', 'wonderful'))
        assert not tag.endswith(('bad', 'awful'))
        text = Text('This ', HRef('/', 'is'), ' good')
        assert not text.endswith('is good')

    def test_isalpha(self):
        assert not HRef('/').isalpha()
        assert not HRef('/', 'a b c').isalpha()
        assert HRef('/', 'abc').isalpha()
        assert HRef('/', u'文字').isalpha()

    def test_render_as(self):
        href = HRef('http://www.example.com', 'Hyperlinked text.')
        assert href.render_as('latex') == '\\href{http://www.example.com}{Hyperlinked text.}'
        assert href.render_as('html') == '<a href="http://www.example.com">Hyperlinked text.</a>'
        assert href.render_as('plaintext') == 'Hyperlinked text.'
        tag = HRef('info.html', Text(), Text('Mary ', 'had ', 'a little lamb'))
        assert tag.render_as('html') == '<a href="info.html">Mary had a little lamb</a>'