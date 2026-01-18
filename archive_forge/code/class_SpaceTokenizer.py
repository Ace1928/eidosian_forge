from nltk.tokenize.api import StringTokenizer, TokenizerI
from nltk.tokenize.util import regexp_span_tokenize, string_span_tokenize
class SpaceTokenizer(StringTokenizer):
    """Tokenize a string using the space character as a delimiter,
    which is the same as ``s.split(' ')``.

        >>> from nltk.tokenize import SpaceTokenizer
        >>> s = "Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\n\\nThanks."
        >>> SpaceTokenizer().tokenize(s) # doctest: +NORMALIZE_WHITESPACE
        ['Good', 'muffins', 'cost', '$3.88\\nin', 'New', 'York.', '',
        'Please', 'buy', 'me\\ntwo', 'of', 'them.\\n\\nThanks.']
    """
    _string = ' '