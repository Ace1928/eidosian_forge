import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class EuroparlCorpusReader(PlaintextCorpusReader):
    """
    Reader for Europarl corpora that consist of plaintext documents.
    Documents are divided into chapters instead of paragraphs as
    for regular plaintext documents. Chapters are separated using blank
    lines. Everything is inherited from ``PlaintextCorpusReader`` except
    that:

    - Since the corpus is pre-processed and pre-tokenized, the
      word tokenizer should just split the line at whitespaces.
    - For the same reason, the sentence tokenizer should just
      split the paragraph at line breaks.
    - There is a new 'chapters()' method that returns chapters instead
      instead of paragraphs.
    - The 'paras()' method inherited from PlaintextCorpusReader is
      made non-functional to remove any confusion between chapters
      and paragraphs for Europarl.
    """

    def _read_word_block(self, stream):
        words = []
        for i in range(20):
            words.extend(stream.readline().split())
        return words

    def _read_sent_block(self, stream):
        sents = []
        for para in self._para_block_reader(stream):
            sents.extend([sent.split() for sent in para.splitlines()])
        return sents

    def _read_para_block(self, stream):
        paras = []
        for para in self._para_block_reader(stream):
            paras.append([sent.split() for sent in para.splitlines()])
        return paras

    def chapters(self, fileids=None):
        """
        :return: the given file(s) as a list of
            chapters, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        return concat([self.CorpusView(fileid, self._read_para_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def paras(self, fileids=None):
        raise NotImplementedError('The Europarl corpus reader does not support paragraphs. Please use chapters() instead.')