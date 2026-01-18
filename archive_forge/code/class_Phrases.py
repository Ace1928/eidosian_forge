import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
from gensim import utils, interfaces
class Phrases(_PhrasesTransformation):
    """Detect phrases based on collocation counts."""

    def __init__(self, sentences=None, min_count=5, threshold=10.0, max_vocab_size=40000000, delimiter='_', progress_per=10000, scoring='default', connector_words=frozenset()):
        """

        Parameters
        ----------
        sentences : iterable of list of str, optional
            The `sentences` iterable can be simply a list, but for larger corpora, consider a generator that streams
            the sentences directly from disk/network, See :class:`~gensim.models.word2vec.BrownCorpus`,
            :class:`~gensim.models.word2vec.Text8Corpus` or :class:`~gensim.models.word2vec.LineSentence`
            for such examples.
        min_count : float, optional
            Ignore all words and bigrams with total collected count lower than this value.
        threshold : float, optional
            Represent a score threshold for forming the phrases (higher means fewer phrases).
            A phrase of words `a` followed by `b` is accepted if the score of the phrase is greater than threshold.
            Heavily depends on concrete scoring-function, see the `scoring` parameter.
        max_vocab_size : int, optional
            Maximum size (number of tokens) of the vocabulary. Used to control pruning of less common words,
            to keep memory under control. The default of 40M needs about 3.6GB of RAM. Increase/decrease
            `max_vocab_size` depending on how much available memory you have.
        delimiter : str, optional
            Glue character used to join collocation tokens.
        scoring : {'default', 'npmi', function}, optional
            Specify how potential phrases are scored. `scoring` can be set with either a string that refers to a
            built-in scoring function, or with a function with the expected parameter names.
            Two built-in scoring functions are available by setting `scoring` to a string:

            #. "default" - :func:`~gensim.models.phrases.original_scorer`.
            #. "npmi" - :func:`~gensim.models.phrases.npmi_scorer`.
        connector_words : set of str, optional
            Set of words that may be included within a phrase, without affecting its scoring.
            No phrase can start nor end with a connector word; a phrase may contain any number of
            connector words in the middle.

            **If your texts are in English, set** ``connector_words=phrases.ENGLISH_CONNECTOR_WORDS``.

            This will cause phrases to include common English articles, prepositions and
            conjuctions, such as `bank_of_america` or `eye_of_the_beholder`.

            For other languages or specific applications domains, use custom ``connector_words``
            that make sense there: ``connector_words=frozenset("der die das".split())`` etc.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Load corpus and train a model.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> # Use the model to detect phrases in a new sentence.
            >>> sent = [u'trees', u'graph', u'minors']
            >>> print(phrases[sent])
            [u'trees_graph', u'minors']
            >>>
            >>> # Or transform multiple sentences at once.
            >>> sents = [[u'trees', u'graph', u'minors'], [u'graph', u'minors']]
            >>> for phrase in phrases[sents]:
            ...     print(phrase)
            [u'trees_graph', u'minors']
            [u'graph_minors']
            >>>
            >>> # Export a FrozenPhrases object that is more efficient but doesn't allow any more training.
            >>> frozen_phrases = phrases.freeze()
            >>> print(frozen_phrases[sent])
            [u'trees_graph', u'minors']

        Notes
        -----

        The ``scoring="npmi"`` is more robust when dealing with common words that form part of common bigrams, and
        ranges from -1 to 1, but is slower to calculate than the default ``scoring="default"``.
        The default is the PMI-like scoring as described in `Mikolov, et. al: "Distributed
        Representations of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

        To use your own custom ``scoring`` function, pass in a function with the following signature:

        * ``worda_count`` - number of corpus occurrences in `sentences` of the first token in the bigram being scored
        * ``wordb_count`` - number of corpus occurrences in `sentences` of the second token in the bigram being scored
        * ``bigram_count`` - number of occurrences in `sentences` of the whole bigram
        * ``len_vocab`` - the number of unique tokens in `sentences`
        * ``min_count`` - the `min_count` setting of the Phrases class
        * ``corpus_word_count`` - the total number of tokens (non-unique) in `sentences`

        The scoring function must accept all these parameters, even if it doesn't use them in its scoring.

        The scoring function **must be pickleable**.

        """
        super().__init__(connector_words=connector_words)
        if min_count <= 0:
            raise ValueError('min_count should be at least 1')
        if threshold <= 0 and scoring == 'default':
            raise ValueError('threshold should be positive for default scoring')
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError('threshold should be between -1 and 1 for npmi scoring')
        if isinstance(scoring, str):
            if scoring == 'default':
                scoring = original_scorer
            elif scoring == 'npmi':
                scoring = npmi_scorer
            else:
                raise ValueError(f'unknown scoring method string {scoring} specified')
        scoring_params = ['worda_count', 'wordb_count', 'bigram_count', 'len_vocab', 'min_count', 'corpus_word_count']
        if callable(scoring):
            missing = [param for param in scoring_params if param not in getargspec(scoring)[0]]
            if not missing:
                self.scoring = scoring
            else:
                raise ValueError(f'scoring function missing expected parameters {missing}')
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.min_reduce = 1
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.corpus_word_count = 0
        try:
            pickle.loads(pickle.dumps(self.scoring))
        except pickle.PickleError:
            raise pickle.PickleError(f'Custom scoring function in {self.__class__.__name__} must be pickle-able')
        if sentences is not None:
            start = time.time()
            self.add_vocab(sentences)
            self.add_lifecycle_event('created', msg=f'built {self} in {time.time() - start:.2f}s')

    def __str__(self):
        return '%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>' % (self.__class__.__name__, len(self.vocab), self.min_count, self.threshold, self.max_vocab_size)

    @staticmethod
    def _learn_vocab(sentences, max_vocab_size, delimiter, connector_words, progress_per):
        """Collect unigram and bigram counts from the `sentences` iterable."""
        sentence_no, total_words, min_reduce = (-1, 0, 1)
        vocab = {}
        logger.info('collecting all words and their counts')
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info('PROGRESS: at sentence #%i, processed %i words and %i word types', sentence_no, total_words, len(vocab))
            start_token, in_between = (None, [])
            for word in sentence:
                if word not in connector_words:
                    vocab[word] = vocab.get(word, 0) + 1
                    if start_token is not None:
                        phrase_tokens = itertools.chain([start_token], in_between, [word])
                        joined_phrase_token = delimiter.join(phrase_tokens)
                        vocab[joined_phrase_token] = vocab.get(joined_phrase_token, 0) + 1
                    start_token, in_between = (word, [])
                elif start_token is not None:
                    in_between.append(word)
                total_words += 1
            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1
        logger.info('collected %i token types (unigram + bigrams) from a corpus of %i words and %i sentences', len(vocab), total_words, sentence_no + 1)
        return (min_reduce, vocab, total_words)

    def add_vocab(self, sentences):
        """Update model parameters with new `sentences`.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus to update this model's parameters from.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Train a phrase detector from a text corpus.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, connector_words=ENGLISH_CONNECTOR_WORDS)  # train model
            >>> assert len(phrases.vocab) == 37
            >>>
            >>> more_sentences = [
            ...     [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],
            ...     [u'machine', u'learning', u'can', u'be', u'new', u'york', u'sometimes'],
            ... ]
            >>>
            >>> phrases.add_vocab(more_sentences)  # add new sentences to model
            >>> assert len(phrases.vocab) == 60

        """
        min_reduce, vocab, total_words = self._learn_vocab(sentences, max_vocab_size=self.max_vocab_size, delimiter=self.delimiter, progress_per=self.progress_per, connector_words=self.connector_words)
        self.corpus_word_count += total_words
        if self.vocab:
            logger.info('merging %i counts into %s', len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in vocab.items():
                self.vocab[word] = self.vocab.get(word, 0) + count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
        else:
            self.vocab = vocab
        logger.info('merged %s', self)

    def score_candidate(self, word_a, word_b, in_between):
        word_a_cnt = self.vocab.get(word_a, 0)
        if word_a_cnt <= 0:
            return (None, None)
        word_b_cnt = self.vocab.get(word_b, 0)
        if word_b_cnt <= 0:
            return (None, None)
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        phrase_cnt = self.vocab.get(phrase, 0)
        if phrase_cnt <= 0:
            return (None, None)
        score = self.scoring(worda_count=word_a_cnt, wordb_count=word_b_cnt, bigram_count=phrase_cnt, len_vocab=len(self.vocab), min_count=self.min_count, corpus_word_count=self.corpus_word_count)
        if score <= self.threshold:
            return (None, None)
        return (phrase, score)

    def freeze(self):
        """
        Return an object that contains the bare minimum of information while still allowing
        phrase detection. See :class:`~gensim.models.phrases.FrozenPhrases`.

        Use this "frozen model" to dramatically reduce RAM footprint if you don't plan to
        make any further changes to your `Phrases` model.

        Returns
        -------
        :class:`~gensim.models.phrases.FrozenPhrases`
            Exported object that's smaller, faster, but doesn't support model updates.

        """
        return FrozenPhrases(self)

    def export_phrases(self):
        """Extract all found phrases.

        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.

        """
        result, source_vocab = ({}, self.vocab)
        for token in source_vocab:
            unigrams = token.split(self.delimiter)
            if len(unigrams) < 2:
                continue
            phrase, score = self.score_candidate(unigrams[0], unigrams[-1], unigrams[1:-1])
            if score is not None:
                result[phrase] = score
        return result