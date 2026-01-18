from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
class SparseTermSimilarityMatrix(SaveLoad):
    """
    Builds a sparse term similarity matrix using a term similarity index.

    Examples
    --------
    >>> from gensim.test.utils import common_texts as corpus, datapath
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import Word2Vec
    >>> from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
    >>> from gensim.similarities.index import AnnoyIndexer
    >>>
    >>> model_corpus_file = datapath('lee_background.cor')
    >>> model = Word2Vec(corpus_file=model_corpus_file, vector_size=20, min_count=1)  # train word-vectors
    >>>
    >>> dictionary = Dictionary(corpus)
    >>> tfidf = TfidfModel(dictionary=dictionary)
    >>> words = [word for word, count in dictionary.most_common()]
    >>> word_vectors = model.wv.vectors_for_all(words, allow_inference=False)  # produce vectors for words in corpus
    >>>
    >>> indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
    >>> termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})
    >>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)  # compute word similarities
    >>>
    >>> tfidf_corpus = tfidf[[dictionary.doc2bow(document) for document in common_texts]]
    >>> docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=10)  # index tfidf_corpus
    >>>
    >>> query = 'graph trees computer'.split()  # make a query
    >>> sims = docsim_index[dictionary.doc2bow(query)]  # find the ten closest documents from tfidf_corpus

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html>`_
    for more examples.

    Parameters
    ----------
    source : :class:`~gensim.similarities.termsim.TermSimilarityIndex` or :class:`scipy.sparse.spmatrix`
        The source of the term similarity. Either a term similarity index that will be used for
        building the term similarity matrix, or an existing sparse term similarity matrix that will
        be encapsulated and stored in the matrix attribute. When a matrix is specified as the
        source, any other parameters will be ignored.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary` or None, optional
        A dictionary that specifies a mapping between terms and the indices of rows and columns
        of the resulting term similarity matrix. The dictionary may only be None when source is
        a :class:`scipy.sparse.spmatrix`.
    tfidf : :class:`gensim.models.tfidfmodel.TfidfModel` or None, optional
        A model that specifies the relative importance of the terms in the dictionary. The columns
        of the term similarity matrix will be build in a decreasing order of importance of
        terms, or in the order of term identifiers if None.
    symmetric : bool, optional
        Whether the symmetry of the term similarity matrix will be enforced. Symmetry is a necessary
        precondition for positive definiteness, which is necessary if you later wish to derive a
        unique change-of-basis matrix from the term similarity matrix using Cholesky factorization.
        Setting symmetric to False will significantly reduce memory usage during matrix construction.
    dominant: bool, optional
        Whether the strict column diagonal dominance of the term similarity matrix will be enforced.
        Strict diagonal dominance and symmetry are sufficient preconditions for positive
        definiteness, which is necessary if you later wish to derive a change-of-basis matrix from
        the term similarity matrix using Cholesky factorization.
    nonzero_limit : int or None, optional
        The maximum number of non-zero elements outside the diagonal in a single column of the
        sparse term similarity matrix. If None, then no limit will be imposed.
    dtype : numpy.dtype, optional
        The data type of the sparse term similarity matrix.

    Attributes
    ----------
    matrix : :class:`scipy.sparse.csc_matrix`
        The encapsulated sparse term similarity matrix.

    Raises
    ------
    ValueError
        If `dictionary` is empty.

    See Also
    --------
    :class:`~gensim.similarities.docsim.SoftCosineSimilarity`
        A document similarity index using the soft cosine similarity over the term similarity matrix.
    :class:`~gensim.similarities.termsim.LevenshteinSimilarityIndex`
        A term similarity index that computes Levenshtein similarities between terms.
    :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`
        A term similarity index that computes cosine similarities between word embeddings.

    """

    def __init__(self, source, dictionary=None, tfidf=None, symmetric=True, dominant=False, nonzero_limit=100, dtype=np.float32):
        if not sparse.issparse(source):
            index = source
            args = (index, dictionary, tfidf, symmetric, dominant, nonzero_limit, dtype)
            source = _create_source(*args)
            assert sparse.issparse(source)
        self.matrix = source.tocsc()

    def inner_product(self, X, Y, normalized=(False, False)):
        """Get the inner product(s) between real vectors / corpora X and Y.

        Return the inner product(s) between real vectors / corpora vec1 and vec2 expressed in a
        non-orthogonal normalized basis, where the dot product between the basis vectors is given by
        the sparse term similarity matrix.

        Parameters
        ----------
        vec1 : list of (int, float) or iterable of list of (int, float)
            A query vector / corpus in the sparse bag-of-words format.
        vec2 : list of (int, float) or iterable of list of (int, float)
            A document vector / corpus in the sparse bag-of-words format.
        normalized : tuple of {True, False, 'maintain'}, optional
            First/second value specifies whether the query/document vectors in the inner product
            will be L2-normalized (True; corresponds to the soft cosine measure), maintain their
            L2-norm during change of basis ('maintain'; corresponds to query expansion with partial
            membership), or kept as-is (False; corresponds to query expansion; default).

        Returns
        -------
        `self.matrix.dtype`,  `scipy.sparse.csr_matrix`, or :class:`numpy.matrix`
            The inner product(s) between `X` and `Y`.

        References
        ----------
        The soft cosine measure was perhaps first described by [sidorovetal14]_.
        Further notes on the efficient implementation of the soft cosine measure are described by
        [novotny18]_.

        .. [sidorovetal14] Grigori Sidorov et al., "Soft Similarity and Soft Cosine Measure: Similarity
           of Features in Vector Space Model", 2014, http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921.

        .. [novotny18] Vít Novotný, "Implementation Notes for the Soft Cosine Measure", 2018,
           http://dx.doi.org/10.1145/3269206.3269317.

        """
        if not X or not Y:
            return self.matrix.dtype.type(0.0)
        normalized_X, normalized_Y = normalized
        valid_normalized_values = (True, False, 'maintain')
        if normalized_X not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_X))
        if normalized_Y not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_Y))
        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)
        if not is_corpus_X and (not is_corpus_Y):
            X = dict(X)
            Y = dict(Y)
            word_indices = np.array(sorted(set(chain(X, Y))))
            dtype = self.matrix.dtype
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = np.array([Y[i] if i in Y else 0 for i in word_indices], dtype=dtype)
            matrix = self.matrix[word_indices[:, None], word_indices].todense()
            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_vector(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)
            return result[0, 0]
        elif not is_corpus_X or not is_corpus_Y:
            if is_corpus_X and (not is_corpus_Y):
                X, Y = (Y, X)
                is_corpus_X, is_corpus_Y = (is_corpus_Y, is_corpus_X)
                normalized_X, normalized_Y = (normalized_Y, normalized_X)
                transposed = True
            else:
                transposed = False
            dtype = self.matrix.dtype
            expanded_X = corpus2csc([X], num_terms=self.matrix.shape[0], dtype=dtype).T.dot(self.matrix)
            word_indices = np.array(sorted(expanded_X.nonzero()[1]))
            del expanded_X
            X = dict(X)
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = corpus2csc(Y, num_terms=self.matrix.shape[0], dtype=dtype)[word_indices, :].todense()
            matrix = self.matrix[word_indices[:, None], word_indices].todense()
            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_corpus(Y, matrix, normalized_Y)
            result = X.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)
            if transposed:
                result = result.T
            return result
        else:
            dtype = self.matrix.dtype
            X = corpus2csc(X if is_corpus_X else [X], num_terms=self.matrix.shape[0], dtype=dtype)
            Y = corpus2csc(Y if is_corpus_Y else [Y], num_terms=self.matrix.shape[0], dtype=dtype)
            matrix = self.matrix
            X = _normalize_sparse_corpus(X, matrix, normalized_X)
            Y = _normalize_sparse_corpus(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result.data = np.clip(result.data, -1.0, 1.0)
            return result