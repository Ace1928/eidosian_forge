import re
import warnings
from typing import Iterator, List, Tuple
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.destructive import MacIntyreContractions
from nltk.tokenize.util import align_tokens
class TreebankWordTokenizer(TokenizerI):
    """
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.

    This tokenizer performs the following steps:

    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line

    >>> from nltk.tokenize import TreebankWordTokenizer
    >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
    >>> TreebankWordTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
    >>> s = "They'll save and invest more."
    >>> TreebankWordTokenizer().tokenize(s)
    ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
    >>> s = "hi, my name can't hello,"
    >>> TreebankWordTokenizer().tokenize(s)
    ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
    """
    STARTING_QUOTES = [(re.compile('^\\"'), '``'), (re.compile('(``)'), ' \\1 '), (re.compile('([ \\(\\[{<])(\\"|\\\'{2})'), '\\1 `` ')]
    PUNCTUATION = [(re.compile('([:,])([^\\d])'), ' \\1 \\2'), (re.compile('([:,])$'), ' \\1 '), (re.compile('\\.\\.\\.'), ' ... '), (re.compile('[;@#$%&]'), ' \\g<0> '), (re.compile('([^\\.])(\\.)([\\]\\)}>"\\\']*)\\s*$'), '\\1 \\2\\3 '), (re.compile('[?!]'), ' \\g<0> '), (re.compile("([^'])' "), "\\1 ' ")]
    PARENS_BRACKETS = (re.compile('[\\]\\[\\(\\)\\{\\}\\<\\>]'), ' \\g<0> ')
    CONVERT_PARENTHESES = [(re.compile('\\('), '-LRB-'), (re.compile('\\)'), '-RRB-'), (re.compile('\\['), '-LSB-'), (re.compile('\\]'), '-RSB-'), (re.compile('\\{'), '-LCB-'), (re.compile('\\}'), '-RCB-')]
    DOUBLE_DASHES = (re.compile('--'), ' -- ')
    ENDING_QUOTES = [(re.compile("''"), " '' "), (re.compile('"'), " '' "), (re.compile("([^' ])('[sS]|'[mM]|'[dD]|') "), '\\1 \\2 '), (re.compile("([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), '\\1 \\2 ')]
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text: str, convert_parentheses: bool=False, return_str: bool=False) -> List[str]:
        """Return a tokenized copy of `text`.

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> s = '''Good muffins cost $3.88 (roughly 3,36 euros)\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
        >>> TreebankWordTokenizer().tokenize(s) # doctest: +NORMALIZE_WHITESPACE
        ['Good', 'muffins', 'cost', '$', '3.88', '(', 'roughly', '3,36',
        'euros', ')', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
        'of', 'them.', 'Thanks', '.']
        >>> TreebankWordTokenizer().tokenize(s, convert_parentheses=True) # doctest: +NORMALIZE_WHITESPACE
        ['Good', 'muffins', 'cost', '$', '3.88', '-LRB-', 'roughly', '3,36',
        'euros', '-RRB-', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
        'of', 'them.', 'Thanks', '.']

        :param text: A string with a sentence or sentences.
        :type text: str
        :param convert_parentheses: if True, replace parentheses to PTB symbols,
            e.g. `(` to `-LRB-`. Defaults to False.
        :type convert_parentheses: bool, optional
        :param return_str: If True, return tokens as space-separated string,
            defaults to False.
        :type return_str: bool, optional
        :return: List of tokens from `text`.
        :rtype: List[str]
        """
        if return_str is not False:
            warnings.warn("Parameter 'return_str' has been deprecated and should no longer be used.", category=DeprecationWarning, stacklevel=2)
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)
        text = ' ' + text + ' '
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(' \\1 \\2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(' \\1 \\2 ', text)
        return text.split()

    def span_tokenize(self, text: str) -> Iterator[Tuple[int, int]]:
        """
        Returns the spans of the tokens in ``text``.
        Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.

            >>> from nltk.tokenize import TreebankWordTokenizer
            >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''
            >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),
            ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),
            ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),
            ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
            >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected
            True
            >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
            ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',
            ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']
            >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
            True

        :param text: A string with a sentence or sentences.
        :type text: str
        :yield: Tuple[int, int]
        """
        raw_tokens = self.tokenize(text)
        if '"' in text or "''" in text:
            matched = [m.group() for m in re.finditer('``|\'{2}|\\"', text)]
            tokens = [matched.pop(0) if tok in ['"', '``', "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        yield from align_tokens(tokens, text)