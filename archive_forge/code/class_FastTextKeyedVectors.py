import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
class FastTextKeyedVectors(KeyedVectors):

    def __init__(self, vector_size, min_n, max_n, bucket, count=0, dtype=REAL):
        """Vectors and vocab for :class:`~gensim.models.fasttext.FastText`.

        Implements significant parts of the FastText algorithm.  For example,
        the :func:`word_vec` calculates vectors for out-of-vocabulary (OOV)
        entities.  FastText achieves this by keeping vectors for ngrams:
        adding the vectors for the ngrams of an entity yields the vector for the
        entity.

        Similar to a hashmap, this class keeps a fixed number of buckets, and
        maps all ngrams to buckets using a hash function.

        Parameters
        ----------
        vector_size : int
            The dimensionality of all vectors.
        min_n : int
            The minimum number of characters in an ngram
        max_n : int
            The maximum number of characters in an ngram
        bucket : int
            The number of buckets.
        count : int, optional
            If provided, vectors will be pre-allocated for at least this many vectors. (Otherwise
            they can be added later.)
        dtype : type, optional
            Vector dimensions will default to `np.float32` (AKA `REAL` in some Gensim code) unless
            another type is provided here.

        Attributes
        ----------
        vectors_vocab : np.array
            Each row corresponds to a vector for an entity in the vocabulary.
            Columns correspond to vector dimensions. When embedded in a full
            FastText model, these are the full-word-token vectors updated
            by training, whereas the inherited vectors are the actual per-word
            vectors synthesized from the full-word-token and all subword (ngram)
            vectors.
        vectors_ngrams : np.array
            A vector for each ngram across all entities in the vocabulary.
            Each row is a vector that corresponds to a bucket.
            Columns correspond to vector dimensions.
        buckets_word : list of np.array
            For each key (by its index), report bucket slots their subwords map to.

        """
        super(FastTextKeyedVectors, self).__init__(vector_size=vector_size, count=count, dtype=dtype)
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.buckets_word = None
        self.vectors_vocab = np.zeros((count, vector_size), dtype=dtype)
        self.vectors_ngrams = None
        self.compatible_hash = True

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        """Load a previously saved `FastTextKeyedVectors` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.fasttext.FastTextKeyedVectors`
            Loaded model.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastTextKeyedVectors.save`
            Save :class:`~gensim.models.fasttext.FastTextKeyedVectors` model.

        """
        return super(FastTextKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(FastTextKeyedVectors, self)._load_specials(*args, **kwargs)
        if not isinstance(self, FastTextKeyedVectors):
            raise TypeError('Loaded object of type %s, not expected FastTextKeyedVectors' % type(self))
        if not hasattr(self, 'compatible_hash') or self.compatible_hash is False:
            raise TypeError('Pre-gensim-3.8.x fastText models with nonstandard hashing are no longer compatible. Loading your old model into gensim-3.8.3 & re-saving may create a model compatible with gensim 4.x.')
        if not hasattr(self, 'vectors_vocab_lockf') and hasattr(self, 'vectors_vocab'):
            self.vectors_vocab_lockf = ones(1, dtype=REAL)
        if not hasattr(self, 'vectors_ngrams_lockf') and hasattr(self, 'vectors_ngrams'):
            self.vectors_ngrams_lockf = ones(1, dtype=REAL)
        if len(self.vectors_vocab_lockf.shape) > 1:
            self.vectors_vocab_lockf = ones(1, dtype=REAL)
        if len(self.vectors_ngrams_lockf.shape) > 1:
            self.vectors_ngrams_lockf = ones(1, dtype=REAL)
        if not hasattr(self, 'buckets_word') or not self.buckets_word:
            self.recalc_char_ngram_buckets()
        if not hasattr(self, 'vectors') or self.vectors is None:
            self.adjust_vectors()

    def __contains__(self, word):
        """Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if current method returns True.

        Parameters
        ----------
        word : str
            Input word.

        Returns
        -------
        bool
            True if `word` or any character ngrams in `word` are present in the vocabulary, False otherwise.

        Note
        ----
        This method **always** returns True with char ngrams, because of the way FastText works.

        If you want to check if a word is an in-vocabulary term, use this instead:

        .. pycon:

            >>> from gensim.test.utils import datapath
            >>> from gensim.models import FastText
            >>> cap_path = datapath("crime-and-punishment.bin")
            >>> model = FastText.load_fasttext_format(cap_path, full_model=False)
            >>> 'steamtrain' in model.wv.key_to_index  # If False, is an OOV term
            False

        """
        if self.bucket == 0:
            return word in self.key_to_index
        else:
            return True

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastTextKeyedVectors.load`
            Load object.

        """
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Arrange any special handling for the gensim.utils.SaveLoad protocol"""
        ignore = set(ignore).union(['buckets_word', 'vectors'])
        return super(FastTextKeyedVectors, self)._save_specials(fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)

    def get_vector(self, word, norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word.
        norm : bool, optional
            If True, resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word and all its ngrams not in vocabulary.

        """
        if word in self.key_to_index:
            return super(FastTextKeyedVectors, self).get_vector(word, norm=norm)
        elif self.bucket == 0:
            raise KeyError('cannot calculate vector for OOV word without ngrams')
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngram_weights = self.vectors_ngrams
            ngram_hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket)
            if len(ngram_hashes) == 0:
                logger.warning('could not extract any ngrams from %r, returning origin vector', word)
                return word_vec
            for nh in ngram_hashes:
                word_vec += ngram_weights[nh]
            if norm:
                return word_vec / np.linalg.norm(word_vec)
            else:
                return word_vec / len(ngram_hashes)

    def get_sentence_vector(self, sentence):
        """Get a single 1-D vector representation for a given `sentence`.
        This function is workalike of the official fasttext's get_sentence_vector().

        Parameters
        ----------
        sentence : list of (str or int)
            list of words specified by string or int ids.

        Returns
        -------
        numpy.ndarray
            1-D numpy array representation of the `sentence`.

        """
        return super(FastTextKeyedVectors, self).get_mean_vector(sentence)

    def resize_vectors(self, seed=0):
        """Make underlying vectors match 'index_to_key' size; random-initialize any new rows."""
        vocab_shape = (len(self.index_to_key), self.vector_size)
        self.vectors_vocab = prep_vectors(vocab_shape, prior_vectors=self.vectors_vocab, seed=seed)
        ngrams_shape = (self.bucket, self.vector_size)
        self.vectors_ngrams = prep_vectors(ngrams_shape, prior_vectors=self.vectors_ngrams, seed=seed + 1)
        self.allocate_vecattrs()
        self.norms = None
        self.recalc_char_ngram_buckets()
        self.adjust_vectors()

    def init_post_load(self, fb_vectors):
        """Perform initialization after loading a native Facebook model.

        Expects that the vocabulary (self.key_to_index) has already been initialized.

        Parameters
        ----------
        fb_vectors : np.array
            A matrix containing vectors for all the entities, including words
            and ngrams.  This comes directly from the binary model.
            The order of the vectors must correspond to the indices in
            the vocabulary.

        """
        vocab_words = len(self)
        assert fb_vectors.shape[0] == vocab_words + self.bucket, 'unexpected number of vectors'
        assert fb_vectors.shape[1] == self.vector_size, 'unexpected vector dimensionality'
        self.vectors_vocab = np.array(fb_vectors[:vocab_words, :])
        self.vectors_ngrams = np.array(fb_vectors[vocab_words:, :])
        self.recalc_char_ngram_buckets()
        self.adjust_vectors()

    def adjust_vectors(self):
        """Adjust the vectors for words in the vocabulary.

        The adjustment composes the trained full-word-token vectors with
        the vectors of the subword ngrams, matching the Facebook reference
        implementation behavior.

        """
        if self.bucket == 0:
            self.vectors = self.vectors_vocab
            return
        self.vectors = self.vectors_vocab[:].copy()
        for i, _ in enumerate(self.index_to_key):
            ngram_buckets = self.buckets_word[i]
            for nh in ngram_buckets:
                self.vectors[i] += self.vectors_ngrams[nh]
            self.vectors[i] /= len(ngram_buckets) + 1

    def recalc_char_ngram_buckets(self):
        """
        Scan the vocabulary, calculate ngrams and their hashes, and cache the list of ngrams for each known word.

        """
        if self.bucket == 0:
            self.buckets_word = [np.array([], dtype=np.uint32)] * len(self.index_to_key)
            return
        self.buckets_word = [None] * len(self.index_to_key)
        for i, word in enumerate(self.index_to_key):
            self.buckets_word[i] = np.array(ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket), dtype=np.uint32)