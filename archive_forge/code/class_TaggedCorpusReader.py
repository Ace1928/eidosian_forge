import os
from nltk.corpus.reader.api import *
from nltk.corpus.reader.timit import read_timit_block
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
from nltk.tokenize import *
class TaggedCorpusReader(CorpusReader):
    """
    Reader for simple part-of-speech tagged corpora.  Paragraphs are
    assumed to be split using blank lines.  Sentences and words can be
    tokenized using the default tokenizers, or by custom tokenizers
    specified as parameters to the constructor.  Words are parsed
    using ``nltk.tag.str2tuple``.  By default, ``'/'`` is used as the
    separator.  I.e., words should have the form::

       word1/tag1 word2/tag2 word3/tag3 ...

    But custom separators may be specified as parameters to the
    constructor.  Part of speech tags are case-normalized to upper
    case.
    """

    def __init__(self, root, fileids, sep='/', word_tokenizer=WhitespaceTokenizer(), sent_tokenizer=RegexpTokenizer('\n', gaps=True), para_block_reader=read_blankline_block, encoding='utf8', tagset=None):
        """
        Construct a new Tagged Corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/...path to corpus.../'
            >>> reader = TaggedCorpusReader(root, '.*', '.txt') # doctest: +SKIP

        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._sep = sep
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader
        self._tagset = tagset

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat([TaggedCorpusView(fileid, enc, False, False, False, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, None) for fileid, enc in self.abspaths(fileids, True)])

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        return concat([TaggedCorpusView(fileid, enc, False, True, False, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, None) for fileid, enc in self.abspaths(fileids, True)])

    def paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        return concat([TaggedCorpusView(fileid, enc, False, True, True, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, None) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_words(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat([TaggedCorpusView(fileid, enc, True, False, False, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, tag_mapping_function) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_sents(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.

        :rtype: list(list(tuple(str,str)))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat([TaggedCorpusView(fileid, enc, True, True, False, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, tag_mapping_function) for fileid, enc in self.abspaths(fileids, True)])

    def tagged_paras(self, fileids=None, tagset=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of ``(word,tag)`` tuples.
        :rtype: list(list(list(tuple(str,str))))
        """
        if tagset and tagset != self._tagset:
            tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
        else:
            tag_mapping_function = None
        return concat([TaggedCorpusView(fileid, enc, True, True, True, self._sep, self._word_tokenizer, self._sent_tokenizer, self._para_block_reader, tag_mapping_function) for fileid, enc in self.abspaths(fileids, True)])