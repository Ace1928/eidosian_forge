import logging
from gensim import utils, matutils
class TransformationABC(utils.SaveLoad):
    """Transformation interface.

    A 'transformation' is any object which accepts document in BoW format via the `__getitem__` (notation `[]`)
    and returns another sparse document in its stead:

    .. sourcecode:: pycon

        >>> from gensim.models import LsiModel
        >>> from gensim.test.utils import common_dictionary, common_corpus
        >>>
        >>> model = LsiModel(common_corpus, id2word=common_dictionary)
        >>> bow_vector = model[common_corpus[0]]  # model applied through __getitem__ on one document from corpus.
        >>> bow_corpus = model[common_corpus]  # also, we can apply model on the full corpus

    """

    def __getitem__(self, vec):
        """Transform a single document, or a whole corpus, from one vector space into another.

        Parameters
        ----------
        vec : {list of (int, number), iterable of list of (int, number)}
            Document in bag-of-words, or streamed corpus.

        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def _apply(self, corpus, chunksize=None, **kwargs):
        """Apply the transformation to a whole corpus and get the result as another corpus.

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in sparse Gensim bag-of-words format.
        chunksize : int, optional
            If provided, a more effective processing will performed.

        Returns
        -------
        :class:`~gensim.interfaces.TransformedCorpus`
            Transformed corpus.

        """
        return TransformedCorpus(self, corpus, chunksize, **kwargs)