from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestTransformStringUsingParseActions(PyparsingExpressionTestCase):
    markup_convert_map = {'*': 'B', '_': 'U', '/': 'I'}

    def markup_convert(t):
        htmltag = TestTransformStringUsingParseActions.markup_convert_map[t.markup_symbol]
        return '<{0}>{1}</{2}>'.format(htmltag, t.body, htmltag)
    tests = [PpTestSpec(desc='Use transformString to convert simple markup to HTML', expr=(pp.oneOf(markup_convert_map)('markup_symbol') + '(' + pp.CharsNotIn(')')('body') + ')').addParseAction(markup_convert), text='Show in *(bold), _(underscore), or /(italic) type', expected_list=['Show in <B>bold</B>, <U>underscore</U>, or <I>italic</I> type'], parse_fn='transformString')]