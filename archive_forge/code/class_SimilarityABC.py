import logging
from gensim import utils, matutils
class SimilarityABC(utils.SaveLoad):
    """Interface for similarity search over a corpus.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.similarities import MatrixSimilarity
        >>> from gensim.test.utils import common_corpus
        >>>
        >>> index = MatrixSimilarity(common_corpus)
        >>> similarities = index.get_similarities(common_corpus[1])  # get similarities between query and corpus

    Notes
    -----
    There is also a convenience wrapper, where iterating over `self` yields similarities of each document in the corpus
    against the whole corpus (i.e. the query is each corpus document in turn).

    See Also
    --------
    :mod:`gensim.similarities`
        Different index implementations of this interface.

    """

    def __init__(self, corpus):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in sparse Gensim bag-of-words format.

        """
        raise NotImplementedError('cannot instantiate Abstract Base Class')

    def get_similarities(self, doc):
        """Get similarities of the given document or corpus against this index.

        Parameters
        ----------
        doc : {list of (int, number), iterable of list of (int, number)}
            Document in the sparse Gensim bag-of-words format, or a streamed corpus of such documents.

        """
        raise NotImplementedError('cannot instantiate Abstract Base Class')

    def __getitem__(self, query):
        """Get similarities of the given document or corpus against this index.

        Uses :meth:`~gensim.interfaces.SimilarityABC.get_similarities` internally.

        Notes
        -----
        Passing an entire corpus as `query` can be more efficient than passing its documents one after another,
        because it will issue queries in batches internally.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number)}
            Document in the sparse Gensim bag-of-words format, or a streamed corpus of such documents.

        Returns
        -------
        {`scipy.sparse.csr.csr_matrix`, list of (int, float)}
            Similarities given document or corpus and objects corpus, depends on `query`.

        """
        is_corpus, query = utils.is_corpus(query)
        if self.normalize:
            if not matutils.ismatrix(query):
                if is_corpus:
                    query = [matutils.unitvec(v) for v in query]
                else:
                    query = matutils.unitvec(query)
        result = self.get_similarities(query)
        if self.num_best is None:
            return result
        if getattr(self, 'maintain_sparsity', False):
            return matutils.scipy2scipy_clipped(result, self.num_best)
        if matutils.ismatrix(result):
            return [matutils.full2sparse_clipped(v, self.num_best) for v in result]
        else:
            return matutils.full2sparse_clipped(result, self.num_best)

    def __iter__(self):
        """Iterate over all documents, compute similarity of each document against all other documents in the index.

        Yields
        ------
        {`scipy.sparse.csr.csr_matrix`, list of (int, float)}
            Similarity of the current document and all documents in the corpus.

        """
        norm = self.normalize
        self.normalize = False
        try:
            chunking = self.chunksize > 1
        except AttributeError:
            chunking = False
        if chunking:
            for chunk_start in range(0, self.index.shape[0], self.chunksize):
                chunk_end = min(self.index.shape[0], chunk_start + self.chunksize)
                chunk = self.index[chunk_start:chunk_end]
                for sim in self[chunk]:
                    yield sim
        else:
            for doc in self.index:
                yield self[doc]
        self.normalize = norm