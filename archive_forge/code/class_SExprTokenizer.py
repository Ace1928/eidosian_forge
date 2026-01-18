import re
from nltk.tokenize.api import TokenizerI
class SExprTokenizer(TokenizerI):
    """
    A tokenizer that divides strings into s-expressions.
    An s-expresion can be either:

      - a parenthesized expression, including any nested parenthesized
        expressions, or
      - a sequence of non-whitespace non-parenthesis characters.

    For example, the string ``(a (b c)) d e (f)`` consists of four
    s-expressions: ``(a (b c))``, ``d``, ``e``, and ``(f)``.

    By default, the characters ``(`` and ``)`` are treated as open and
    close parentheses, but alternative strings may be specified.

    :param parens: A two-element sequence specifying the open and close parentheses
        that should be used to find sexprs.  This will typically be either a
        two-character string, or a list of two strings.
    :type parens: str or list
    :param strict: If true, then raise an exception when tokenizing an ill-formed sexpr.
    """

    def __init__(self, parens='()', strict=True):
        if len(parens) != 2:
            raise ValueError('parens must contain exactly two strings')
        self._strict = strict
        self._open_paren = parens[0]
        self._close_paren = parens[1]
        self._paren_regexp = re.compile(f'{re.escape(parens[0])}|{re.escape(parens[1])}')

    def tokenize(self, text):
        """
        Return a list of s-expressions extracted from *text*.
        For example:

            >>> SExprTokenizer().tokenize('(a b (c d)) e f (g)')
            ['(a b (c d))', 'e', 'f', '(g)']

        All parentheses are assumed to mark s-expressions.
        (No special processing is done to exclude parentheses that occur
        inside strings, or following backslash characters.)

        If the given expression contains non-matching parentheses,
        then the behavior of the tokenizer depends on the ``strict``
        parameter to the constructor.  If ``strict`` is ``True``, then
        raise a ``ValueError``.  If ``strict`` is ``False``, then any
        unmatched close parentheses will be listed as their own
        s-expression; and the last partial s-expression with unmatched open
        parentheses will be listed as its own s-expression:

            >>> SExprTokenizer(strict=False).tokenize('c) d) e (f (g')
            ['c', ')', 'd', ')', 'e', '(f (g']

        :param text: the string to be tokenized
        :type text: str or iter(str)
        :rtype: iter(str)
        """
        result = []
        pos = 0
        depth = 0
        for m in self._paren_regexp.finditer(text):
            paren = m.group()
            if depth == 0:
                result += text[pos:m.start()].split()
                pos = m.start()
            if paren == self._open_paren:
                depth += 1
            if paren == self._close_paren:
                if self._strict and depth == 0:
                    raise ValueError('Un-matched close paren at char %d' % m.start())
                depth = max(0, depth - 1)
                if depth == 0:
                    result.append(text[pos:m.end()])
                    pos = m.end()
        if self._strict and depth > 0:
            raise ValueError('Un-matched open paren at char %d' % pos)
        if pos < len(text):
            result.append(text[pos:])
        return result