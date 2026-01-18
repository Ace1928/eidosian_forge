import logging
from gensim import utils, matutils
class TransformedCorpus(CorpusABC):
    """Interface for corpora that are the result of an online (streamed) transformation."""

    def __init__(self, obj, corpus, chunksize=None, **kwargs):
        """

        Parameters
        ----------
        obj : object
            A transformation :class:`~gensim.interfaces.TransformationABC` object that will be applied
            to each document from `corpus` during iteration.
        corpus : iterable of list of (int, number)
            Corpus in bag-of-words format.
        chunksize : int, optional
            If provided, a slightly more effective processing will be performed by grouping documents from `corpus`.

        """
        self.obj, self.corpus, self.chunksize = (obj, corpus, chunksize)
        for key, value in kwargs.items():
            setattr(self.obj, key, value)
        self.metadata = False

    def __len__(self):
        """Get corpus size."""
        return len(self.corpus)

    def __iter__(self):
        """Iterate over the corpus, applying the selected transformation.

        If `chunksize` was set in the constructor, works in "batch-manner" (more efficient).

        Yields
        ------
        list of (int, number)
            Documents in the sparse Gensim bag-of-words format.

        """
        if self.chunksize:
            for chunk in utils.grouper(self.corpus, self.chunksize):
                for transformed in self.obj.__getitem__(chunk, chunksize=None):
                    yield transformed
        else:
            for doc in self.corpus:
                yield self.obj[doc]

    def __getitem__(self, docno):
        """Transform the document at position `docno` within `corpus` specified in the constructor.

        Parameters
        ----------
        docno : int
            Position of the document to transform. Document offset inside `self.corpus`.

        Notes
        -----
        `self.corpus` must support random indexing.

        Returns
        -------
        list of (int, number)
            Transformed document in the sparse Gensim bag-of-words format.

        Raises
        ------
        RuntimeError
            If corpus doesn't support index slicing (`__getitem__` doesn't exists).

        """
        if hasattr(self.corpus, '__getitem__'):
            return self.obj[self.corpus[docno]]
        else:
            raise RuntimeError('Type {} does not support slicing.'.format(type(self.corpus)))