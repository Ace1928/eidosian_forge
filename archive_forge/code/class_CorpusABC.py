import logging
from gensim import utils, matutils
class CorpusABC(utils.SaveLoad):
    """Interface for corpus classes from :mod:`gensim.corpora`.

    Corpus is simply an iterable object, where each iteration step yields one document:

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus  # inherits from the CorpusABC class
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath("testcorpus.mm"))
        >>> for doc in corpus:
        ...     pass  # do something with the doc...

    A document represented in the bag-of-word (BoW) format, i.e. list of (attr_id, attr_value),
    like ``[(1, 0.2), (4, 0.6), ...]``.

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus  # inherits from the CorpusABC class
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath("testcorpus.mm"))
        >>> doc = next(iter(corpus))
        >>> print(doc)
        [(0, 1.0), (1, 1.0), (2, 1.0)]

    Remember that the save/load methods only pickle the corpus object, not
    the (streamed) corpus data itself!
    To save the corpus data, please use this pattern :

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus  # MmCorpus inherits from CorpusABC
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>>
        >>> corpus = MmCorpus(datapath("testcorpus.mm"))
        >>> tmp_path = get_tmpfile("temp_corpus.mm")
        >>>
        >>> MmCorpus.serialize(tmp_path, corpus)  # serialize corpus to disk in the MmCorpus format
        >>> loaded_corpus = MmCorpus(tmp_path)  # load corpus through constructor
        >>> for (doc_1, doc_2) in zip(corpus, loaded_corpus):
        ...     assert doc_1 == doc_2  # no change between the original and loaded corpus


    See Also
    --------
    :mod:`gensim.corpora`
        Corpora in different formats.

    """

    def __iter__(self):
        """Iterate all over corpus."""
        raise NotImplementedError('cannot instantiate abstract base class')

    def save(self, *args, **kwargs):
        """Saves the in-memory state of the corpus (pickles the object).

        Warnings
        --------
        This saves only the "internal state" of the corpus object, not the corpus data!

        To save the corpus data, use the `serialize` method of your desired output format
        instead, e.g. :meth:`gensim.corpora.mmcorpus.MmCorpus.serialize`.

        """
        import warnings
        warnings.warn('corpus.save() stores only the (tiny) iteration object in memory; to serialize the actual corpus content, use e.g. MmCorpus.serialize(corpus)')
        super(CorpusABC, self).save(*args, **kwargs)

    def __len__(self):
        """Get the corpus size = the total number of documents in it."""
        raise NotImplementedError('must override __len__() before calling len(corpus)')

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Save `corpus` to disk.

        Some formats support saving the dictionary (`feature_id -> word` mapping),
        which can be provided by the optional `id2word` parameter.

        Notes
        -----
        Some corpora also support random access via document indexing, so that the documents on disk
        can be accessed in O(1) time (see the :class:`gensim.corpora.indexedcorpus.IndexedCorpus` base class).

        In this case, :meth:`~gensim.interfaces.CorpusABC.save_corpus` is automatically called internally by
        :func:`serialize`, which does :meth:`~gensim.interfaces.CorpusABC.save_corpus` plus saves the index
        at the same time.

        Calling :func:`serialize() is preferred to calling :meth:`gensim.interfaces.CorpusABC.save_corpus`.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        id2word : :class:`~gensim.corpora.Dictionary`, optional
            Dictionary of corpus.
        metadata : bool, optional
            Write additional metadata to a separate too?

        """
        raise NotImplementedError('cannot instantiate abstract base class')