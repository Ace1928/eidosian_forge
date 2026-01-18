import logging
import numpy as np
from gensim import interfaces, matutils, utils
class RpModel(interfaces.TransformationABC):

    def __init__(self, corpus, id2word=None, num_topics=300):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.

        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping `token_id` -> `token`, will be determine from corpus if `id2word == None`.

        num_topics : int, optional
            Number of topics.

        """
        self.id2word = id2word
        self.num_topics = num_topics
        if corpus is not None:
            self.initialize(corpus)
            self.add_lifecycle_event('created', msg=f'created {self}')

    def __str__(self):
        return '%s<num_terms=%s, num_topics=%s>' % (self.__class__.__name__, self.num_terms, self.num_topics)

    def initialize(self, corpus):
        """Initialize the random projection matrix.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
          Input corpus.

        """
        if self.id2word is None:
            logger.info('no word id mapping provided; initializing from corpus, assuming identity')
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif self.id2word:
            self.num_terms = 1 + max(self.id2word)
        else:
            self.num_terms = 0
        shape = (self.num_topics, self.num_terms)
        logger.info('constructing %s random matrix', str(shape))
        randmat = 1 - 2 * np.random.binomial(1, 0.5, shape)
        self.projection = np.asfortranarray(randmat, dtype=np.float32)

    def __getitem__(self, bow):
        """Get random-projection representation of the input vector or corpus.

        Parameters
        ----------
        bow : {list of (int, int), iterable of list of (int, int)}
            Input document or corpus.

        Returns
        -------
        list of (int, float)
            if `bow` is document OR
        :class:`~gensim.interfaces.TransformedCorpus`
            if `bow` is corpus.

        Examples
        ----------
        .. sourcecode:: pycon

            >>> from gensim.models import RpModel
            >>> from gensim.corpora import Dictionary
            >>> from gensim.test.utils import common_texts
            >>>
            >>> dictionary = Dictionary(common_texts)  # fit dictionary
            >>> corpus = [dictionary.doc2bow(text) for text in common_texts]  # convert texts to BoW format
            >>>
            >>> model = RpModel(corpus, id2word=dictionary)  # fit model
            >>>
            >>> # apply model to document, result is vector in BoW format, i.e. [(1, 0.3), ... ]
            >>> result = model[corpus[0]]

        """
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)
        if getattr(self, 'freshly_loaded', False):
            self.freshly_loaded = False
            self.projection = self.projection.copy('F')
        vec = matutils.sparse2full(bow, self.num_terms).reshape(self.num_terms, 1) / np.sqrt(self.num_topics)
        vec = np.asfortranarray(vec, dtype=np.float32)
        topic_dist = np.dot(self.projection, vec)
        return [(topicid, float(topicvalue)) for topicid, topicvalue in enumerate(topic_dist.flat) if np.isfinite(topicvalue) and (not np.allclose(topicvalue, 0.0))]

    def __setstate__(self, state):
        """Sets the internal state and updates freshly_loaded to True, called when unpicked.

        Parameters
        ----------
        state : dict
           State of the class.

        """
        self.__dict__ = state
        self.freshly_loaded = True