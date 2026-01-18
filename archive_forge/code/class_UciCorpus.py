import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
class UciCorpus(UciReader, IndexedCorpus):
    """Corpus in the UCI bag-of-words format."""

    def __init__(self, fname, fname_vocab=None):
        """
        Parameters
        ----------
        fname : str
            Path to corpus in UCI format.
        fname_vocab : bool, optional
            Path to vocab.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import UciCorpus
            >>> from gensim.test.utils import datapath
            >>>
            >>> corpus = UciCorpus(datapath('testcorpus.uci'))
            >>> for document in corpus:
            ...     pass

        """
        IndexedCorpus.__init__(self, fname)
        UciReader.__init__(self, fname)
        if fname_vocab is None:
            fname_vocab = utils.smart_extension(fname, '.vocab')
        self.fname = fname
        with utils.open(fname_vocab, 'rb') as fin:
            words = [word.strip() for word in fin]
        self.id2word = dict(enumerate(words))
        self.transposed = True

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format.

        """
        for docId, doc in super(UciCorpus, self).__iter__():
            yield doc

    def create_dictionary(self):
        """Generate :class:`gensim.corpora.dictionary.Dictionary` directly from the corpus and vocabulary data.

        Return
        ------
        :class:`gensim.corpora.dictionary.Dictionary`
            Dictionary, based on corpus.

        Examples
        --------

        .. sourcecode:: pycon

            >>> from gensim.corpora.ucicorpus import UciCorpus
            >>> from gensim.test.utils import datapath
            >>> ucc = UciCorpus(datapath('testcorpus.uci'))
            >>> dictionary = ucc.create_dictionary()

        """
        dictionary = Dictionary()
        dictionary.dfs = defaultdict(int)
        dictionary.id2token = self.id2word
        dictionary.token2id = utils.revdict(self.id2word)
        dictionary.num_docs = self.num_docs
        dictionary.num_nnz = self.num_nnz
        for docno, doc in enumerate(self):
            if docno % 10000 == 0:
                logger.info('PROGRESS: processing document %i of %i', docno, self.num_docs)
            for word, count in doc:
                dictionary.dfs[word] += 1
                dictionary.num_pos += count
        return dictionary

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=10000, metadata=False):
        """Save a corpus in the UCI Bag-of-Words format.

        Warnings
        --------
        This function is automatically called by :meth`gensim.corpora.ucicorpus.UciCorpus.serialize`,
        don't call it directly, call :meth`gensim.corpora.ucicorpus.UciCorpus.serialize` instead.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus: iterable of iterable of (int, int)
            Corpus in BoW format.
        id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between words and their ids. If None - will be inferred from `corpus`.
        progress_cnt : int, optional
            Progress counter, write log message each `progress_cnt` documents.
        metadata : bool, optional
            THIS PARAMETER WILL BE IGNORED.

        Notes
        -----
        There are actually two files saved: `fname` and `fname.vocab`, where `fname.vocab` is the vocabulary file.

        """
        if id2word is None:
            logger.info('no word id mapping provided; initializing from corpus')
            id2word = utils.dict_from_corpus(corpus)
            num_terms = len(id2word)
        elif id2word:
            num_terms = 1 + max(id2word)
        else:
            num_terms = 0
        fname_vocab = utils.smart_extension(fname, '.vocab')
        logger.info('saving vocabulary of %i words to %s', num_terms, fname_vocab)
        with utils.open(fname_vocab, 'wb') as fout:
            for featureid in range(num_terms):
                fout.write(utils.to_utf8('%s\n' % id2word.get(featureid, '---')))
        logger.info('storing corpus in UCI Bag-of-Words format: %s', fname)
        return UciWriter.write_corpus(fname, corpus, index=True, progress_cnt=progress_cnt)