from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text

    Parses a string into a kind of AST

        >>> parse('{{x}}')
        [('expr', (1, 3), 'x')]
        >>> parse('foo')
        ['foo']
        >>> parse('{{if x}}test{{endif}}')
        [('cond', (1, 3), ('if', (1, 3), 'x', ['test']))]
        >>> parse('series->{{for x in y}}x={{x}}{{endfor}}')
        ['series->', ('for', (1, 11), ('x',), 'y', ['x=', ('expr', (1, 27), 'x')])]
        >>> parse('{{for x, y in z:}}{{continue}}{{endfor}}')
        [('for', (1, 3), ('x', 'y'), 'z', [('continue', (1, 21))])]
        >>> parse('{{py:x=1}}')
        [('py', (1, 3), 'x=1')]
        >>> parse('{{if x}}a{{elif y}}b{{else}}c{{endif}}')
        [('cond', (1, 3), ('if', (1, 3), 'x', ['a']), ('elif', (1, 12), 'y', ['b']), ('else', (1, 23), None, ['c']))]

    Some exceptions::

        >>> parse('{{continue}}')
        Traceback (most recent call last):
            ...
        TemplateError: continue outside of for loop at line 1 column 3
        >>> parse('{{if x}}foo')
        Traceback (most recent call last):
            ...
        TemplateError: No {{endif}} at line 1 column 3
        >>> parse('{{else}}')
        Traceback (most recent call last):
            ...
        TemplateError: else outside of an if block at line 1 column 3
        >>> parse('{{if x}}{{for x in y}}{{endif}}{{endfor}}')
        Traceback (most recent call last):
            ...
        TemplateError: Unexpected endif at line 1 column 25
        >>> parse('{{if}}{{endif}}')
        Traceback (most recent call last):
            ...
        TemplateError: if with no expression at line 1 column 3
        >>> parse('{{for x y}}{{endfor}}')
        Traceback (most recent call last):
            ...
        TemplateError: Bad for (no "in") in 'x y' at line 1 column 3
        >>> parse('{{py:x=1\ny=2}}')
        Traceback (most recent call last):
            ...
        TemplateError: Multi-line py blocks must start with a newline at line 1 column 3
    