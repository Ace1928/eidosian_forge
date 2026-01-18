from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus.reader.api import *
class OpinionLexiconCorpusReader(WordListCorpusReader):
    """
    Reader for Liu and Hu opinion lexicon.  Blank lines and readme are ignored.

        >>> from nltk.corpus import opinion_lexicon
        >>> opinion_lexicon.words()
        ['2-faced', '2-faces', 'abnormal', 'abolish', ...]

    The OpinionLexiconCorpusReader provides shortcuts to retrieve positive/negative
    words:

        >>> opinion_lexicon.negative()
        ['2-faced', '2-faces', 'abnormal', 'abolish', ...]

    Note that words from `words()` method are sorted by file id, not alphabetically:

        >>> opinion_lexicon.words()[0:10] # doctest: +NORMALIZE_WHITESPACE
        ['2-faced', '2-faces', 'abnormal', 'abolish', 'abominable', 'abominably',
        'abominate', 'abomination', 'abort', 'aborted']
        >>> sorted(opinion_lexicon.words())[0:10] # doctest: +NORMALIZE_WHITESPACE
        ['2-faced', '2-faces', 'a+', 'abnormal', 'abolish', 'abominable', 'abominably',
        'abominate', 'abomination', 'abort']
    """
    CorpusView = IgnoreReadmeCorpusView

    def words(self, fileids=None):
        """
        Return all words in the opinion lexicon. Note that these words are not
        sorted in alphabetical order.

        :param fileids: a list or regexp specifying the ids of the files whose
            words have to be returned.
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.CorpusView(path, self._read_word_block, encoding=enc) for path, enc, fileid in self.abspaths(fileids, True, True)])

    def positive(self):
        """
        Return all positive words in alphabetical order.

        :return: a list of positive words.
        :rtype: list(str)
        """
        return self.words('positive-words.txt')

    def negative(self):
        """
        Return all negative words in alphabetical order.

        :return: a list of negative words.
        :rtype: list(str)
        """
        return self.words('negative-words.txt')

    def _read_word_block(self, stream):
        words = []
        for i in range(20):
            line = stream.readline()
            if not line:
                continue
            words.append(line.strip())
        return words