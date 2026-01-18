import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
class FastText(Word2Vec):

    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, word_ngrams=1, sample=0.001, seed=1, workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(), max_final_vocab=None, shrink_windows=True):
        """Train, use and evaluate word representations learned using the method
        described in `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_,
        aka FastText.

        The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save` and
        :meth:`~gensim.models.fasttext.FastText.load` methods, or loaded from a format compatible with the
        original Fasttext implementation via :func:`~gensim.models.fasttext.load_facebook_model`.

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus'
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such
            examples. If you don't supply `sentences`, the model is left uninitialized -- use if you plan to
            initialize it in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left
            uninitialized).
        min_count : int, optional
            The model ignores all words with total frequency lower than this.
        vector_size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : {1, 0}, optional
            Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of themodel.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        sorted_vocab : {1,0}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indices.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        min_n : int, optional
            Minimum length of char n-grams to be used for training word representations.
        max_n : int, optional
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : int, optional
            In Facebook's FastText, "max length of word ngram" - but gensim only supports the
            default of 1 (regular unigram word handling).
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
            The default value of 2000000 consumes as much memory as having 2000000 more in-vocabulary
            words in your model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.
        max_final_vocab : int, optional
            Limits the vocab to a target vocab size by automatically selecting
            ``min_count```.  If the specified ``min_count`` is more than the
            automatically calculated ``min_count``, the former will be used.
            Set to ``None`` if not required.
        shrink_windows : bool, optional
            New in 4.1. Experimental.
            If True, the effective window size is uniformly sampled from  [1, `window`]
            for each target word during training, to match the original word2vec algorithm's
            approximate weighting of context words by distance. Otherwise, the effective
            window size is always fixed to `window` words to either side.

        Examples
        --------
        Initialize and train a `FastText` model:

        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = FastText(sentences, min_count=1)
            >>> say_vector = model.wv['say']  # get vector for word
            >>> of_vector = model.wv['of']  # get vector for out-of-vocab word

        Attributes
        ----------
        wv : :class:`~gensim.models.fasttext.FastTextKeyedVectors`
            This object essentially contains the mapping between words and embeddings. These are similar to
            the embedding computed in the :class:`~gensim.models.word2vec.Word2Vec`, however here we also
            include vectors for n-grams. This allows the model to compute embeddings even for **unseen**
            words (that do not exist in the vocabulary), as the aggregate of the n-grams included in the word.
            After training the model, this attribute can be used directly to query those embeddings in various
            ways. Check the module level docstring for some examples.

        """
        self.load = utils.call_on_class_only
        self.load_fasttext_format = utils.call_on_class_only
        self.callbacks = callbacks
        if word_ngrams != 1:
            raise NotImplementedError("Gensim's FastText implementation does not yet support word_ngrams != 1.")
        self.word_ngrams = word_ngrams
        if max_n < min_n:
            bucket = 0
        self.wv = FastTextKeyedVectors(vector_size, min_n, max_n, bucket)
        self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)
        super(FastText, self).__init__(sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=vector_size, epochs=epochs, callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window, max_vocab_size=max_vocab_size, max_final_vocab=max_final_vocab, min_count=min_count, sample=sample, sorted_vocab=sorted_vocab, null_word=null_word, ns_exponent=ns_exponent, hashfxn=hashfxn, seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, shrink_windows=shrink_windows)

    def _init_post_load(self, hidden_output):
        num_vectors = len(self.wv.vectors)
        vocab_size = len(self.wv)
        vector_size = self.wv.vector_size
        assert num_vectors > 0, 'expected num_vectors to be initialized already'
        assert vocab_size > 0, 'expected vocab_size to be initialized already'
        self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)
        self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
        if self.hs:
            self.syn1 = hidden_output
        if self.negative:
            self.syn1neg = hidden_output
        self.layer1_size = vector_size

    def _clear_post_train(self):
        """Clear any cached values that training may have invalidated."""
        super(FastText, self)._clear_post_train()
        self.wv.adjust_vectors()

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate memory that will be needed to train a model, and print the estimates to log."""
        vocab_size = vocab_size or len(self.wv)
        vec_size = self.vector_size * np.dtype(np.float32).itemsize
        l1_size = self.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report['vocab'] = len(self.wv) * (700 if self.hs else 500)
        report['syn0_vocab'] = len(self.wv) * vec_size
        num_buckets = self.wv.bucket
        if self.hs:
            report['syn1'] = len(self.wv) * l1_size
        if self.negative:
            report['syn1neg'] = len(self.wv) * l1_size
        if self.wv.bucket:
            report['syn0_ngrams'] = self.wv.bucket * vec_size
            num_ngrams = 0
            for word in self.wv.key_to_index:
                hashes = ft_ngram_hashes(word, self.wv.min_n, self.wv.max_n, self.wv.bucket)
                num_ngrams += len(hashes)
            report['buckets_word'] = 64 + 100 * len(self.wv) + 4 * num_ngrams
        report['total'] = sum(report.values())
        logger.info('estimated required memory for %i words, %i buckets and %i dimensions: %i bytes', len(self.wv), num_buckets, self.vector_size, report['total'])
        return report

    def _do_train_epoch(self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch, total_examples=None, total_words=None, **kwargs):
        work, neu1 = thread_private_mem
        if self.sg:
            examples, tally, raw_tally = train_epoch_sg(self, corpus_file, offset, cython_vocab, cur_epoch, total_examples, total_words, work, neu1)
        else:
            examples, tally, raw_tally = train_epoch_cbow(self, corpus_file, offset, cython_vocab, cur_epoch, total_examples, total_words, work, neu1)
        return (examples, tally, raw_tally)

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of list of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : tuple of (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Each worker's private work memory.

        Returns
        -------
        (int, int)
            Tuple of (effective word count after ignoring unknown words and sentence length trimming, total word count)

        """
        work, neu1 = inits
        tally = train_batch_any(self, sentences, alpha, work, neu1)
        return (tally, self._raw_word_count(sentences))

    @deprecated('Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. init_sims() is now obsoleted and will be completely removed in future versions. See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4')
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors. Obsoleted.

        If you need a single unit-normalized vector for some key, call
        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:
        ``fasttext_model.wv.get_vector(key, norm=True)``.

        To refresh norms after you performed some atypical out-of-band vector tampering,
        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.

        Parameters
        ----------
        replace : bool
            If True, forget the original trained vectors and only keep the normalized ones.
            You lose information if you do this.

        """
        self.wv.init_sims(replace=replace)

    @classmethod
    @utils.deprecated('use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model (to continue training with the loaded full model, more RAM) instead')
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """Deprecated.

        Use :func:`gensim.models.fasttext.load_facebook_model` or
        :func:`gensim.models.fasttext.load_facebook_vectors` instead.

        """
        return load_facebook_model(model_file, encoding=encoding)

    @utils.deprecated('use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model (to continue training with the loaded full model, more RAM) instead')
    def load_binary_data(self, encoding='utf8'):
        """Load data from a binary file created by Facebook's native FastText.

        Parameters
        ----------
        encoding : str, optional
            Specifies the encoding.

        """
        m = _load_fasttext_format(self.file_name, encoding=encoding)
        for attr, val in m.__dict__.items():
            setattr(self, attr, val)

    def save(self, *args, **kwargs):
        """Save the Fasttext model. This saved model can be loaded again using
        :meth:`~gensim.models.fasttext.FastText.load`, which supports incremental training
        and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Store the model to this file.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.load`
            Load :class:`~gensim.models.fasttext.FastText` model.

        """
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved `FastText` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.fasttext.FastText`
            Loaded model.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.save`
            Save :class:`~gensim.models.fasttext.FastText` model.

        """
        return super(FastText, cls).load(*args, rethrow=True, **kwargs)

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(FastText, self)._load_specials(*args, **kwargs)
        if hasattr(self, 'bucket'):
            self.wv.bucket = self.bucket
            del self.bucket