from nltk.tokenize.api import StringTokenizer, TokenizerI
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize
class LineTokenizer(TokenizerI):
    """Tokenize a string into its lines, optionally discarding blank lines.
    This is similar to ``s.split('\\n')``.

        >>> from nltk.tokenize import LineTokenizer
        >>> s = "Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\n\\nThanks."
        >>> LineTokenizer(blanklines='keep').tokenize(s) # doctest: +NORMALIZE_WHITESPACE
        ['Good muffins cost $3.88', 'in New York.  Please buy me',
        'two of them.', '', 'Thanks.']
        >>> # same as [l for l in s.split('\\n') if l.strip()]:
        >>> LineTokenizer(blanklines='discard').tokenize(s) # doctest: +NORMALIZE_WHITESPACE
        ['Good muffins cost $3.88', 'in New York.  Please buy me',
        'two of them.', 'Thanks.']

    :param blanklines: Indicates how blank lines should be handled.  Valid values are:

        - ``discard``: strip blank lines out of the token list before returning it.
           A line is considered blank if it contains only whitespace characters.
        - ``keep``: leave all blank lines in the token list.
        - ``discard-eof``: if the string ends with a newline, then do not generate
           a corresponding token ``''`` after that newline.
    """

    def __init__(self, blanklines='discard'):
        valid_blanklines = ('discard', 'keep', 'discard-eof')
        if blanklines not in valid_blanklines:
            raise ValueError('Blank lines must be one of: %s' % ' '.join(valid_blanklines))
        self._blanklines = blanklines

    def tokenize(self, s):
        lines = s.splitlines()
        if self._blanklines == 'discard':
            lines = [l for l in lines if l.rstrip()]
        elif self._blanklines == 'discard-eof':
            if lines and (not lines[-1].strip()):
                lines.pop()
        return lines

    def span_tokenize(self, s):
        if self._blanklines == 'keep':
            yield from string_span_tokenize(s, '\\n')
        else:
            yield from regexp_span_tokenize(s, '\\n(\\s+\\n)*')