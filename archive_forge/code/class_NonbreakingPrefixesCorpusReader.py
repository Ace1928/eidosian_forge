from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import line_tokenize
class NonbreakingPrefixesCorpusReader(WordListCorpusReader):
    """
    This is a class to read the nonbreaking prefixes textfiles from the
    Moses Machine Translation toolkit. These lists are used in the Python port
    of the Moses' word tokenizer.
    """
    available_langs = {'catalan': 'ca', 'czech': 'cs', 'german': 'de', 'greek': 'el', 'english': 'en', 'spanish': 'es', 'finnish': 'fi', 'french': 'fr', 'hungarian': 'hu', 'icelandic': 'is', 'italian': 'it', 'latvian': 'lv', 'dutch': 'nl', 'polish': 'pl', 'portuguese': 'pt', 'romanian': 'ro', 'russian': 'ru', 'slovak': 'sk', 'slovenian': 'sl', 'swedish': 'sv', 'tamil': 'ta'}
    available_langs.update({v: v for v in available_langs.values()})

    def words(self, lang=None, fileids=None, ignore_lines_startswith='#'):
        """
        This module returns a list of nonbreaking prefixes for the specified
        language(s).

        >>> from nltk.corpus import nonbreaking_prefixes as nbp
        >>> nbp.words('en')[:10] == [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J']
        True
        >>> nbp.words('ta')[:5] == [u'அ', u'ஆ', u'இ', u'ஈ', u'உ']
        True

        :return: a list words for the specified language(s).
        """
        if lang in self.available_langs:
            lang = self.available_langs[lang]
            fileids = ['nonbreaking_prefix.' + lang]
        return [line for line in line_tokenize(self.raw(fileids)) if not line.startswith(ignore_lines_startswith)]