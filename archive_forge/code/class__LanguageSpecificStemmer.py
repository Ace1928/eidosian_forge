import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class _LanguageSpecificStemmer(StemmerI):
    """
    This helper subclass offers the possibility
    to invoke a specific stemmer directly.
    This is useful if you already know the language to be stemmed at runtime.

    Create an instance of the Snowball stemmer.

    :param ignore_stopwords: If set to True, stopwords are
                             not stemmed and returned unchanged.
                             Set to False by default.
    :type ignore_stopwords: bool
    """

    def __init__(self, ignore_stopwords=False):
        language = type(self).__name__.lower()
        if language.endswith('stemmer'):
            language = language[:-7]
        self.stopwords = set()
        if ignore_stopwords:
            try:
                for word in stopwords.words(language):
                    self.stopwords.add(word)
            except OSError as e:
                raise ValueError("{!r} has no list of stopwords. Please set 'ignore_stopwords' to 'False'.".format(self)) from e

    def __repr__(self):
        """
        Print out the string representation of the respective class.

        """
        return f'<{type(self).__name__}>'