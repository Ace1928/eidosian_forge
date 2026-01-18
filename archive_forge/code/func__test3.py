import re
import sys
from pprint import pprint
def _test3():
    """
    >>> vtor.check('string(default="")', '', missing=True)
    ''
    >>> vtor.check('string(default="\\n")', '', missing=True)
    '\\n'
    >>> print(vtor.check('string(default="\\n")', '', missing=True))
    <BLANKLINE>
    <BLANKLINE>
    >>> vtor.check('string()', '\\n')
    '\\n'
    >>> vtor.check('string(default="\\n\\n\\n")', '', missing=True)
    '\\n\\n\\n'
    >>> vtor.check('string()', 'random \\n text goes here\\n\\n')
    'random \\n text goes here\\n\\n'
    >>> vtor.check('string(default=" \\nrandom text\\ngoes \\n here\\n\\n ")',
    ... '', missing=True)
    ' \\nrandom text\\ngoes \\n here\\n\\n '
    >>> vtor.check("string(default='\\n\\n\\n')", '', missing=True)
    '\\n\\n\\n'
    >>> vtor.check("option('\\n','a','b',default='\\n')", '', missing=True)
    '\\n'
    >>> vtor.check("string_list()", ['foo', '\\n', 'bar'])
    ['foo', '\\n', 'bar']
    >>> vtor.check("string_list(default=list('\\n'))", '', missing=True)
    ['\\n']
    """