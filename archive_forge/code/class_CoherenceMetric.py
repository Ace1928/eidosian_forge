import gensim
import logging
import copy
import sys
import numpy as np
class CoherenceMetric(Metric):
    """Metric class for coherence evaluation.

    See Also
    --------
    :class:`~gensim.models.coherencemodel.CoherenceModel`

    """

    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None, window_size=None, topn=10, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        texts : list of char (str of length 1), optional
            Tokenized texts needed for coherence models that use sliding window based probability estimator.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping from integer IDs to words, needed to create corpus. If `model.id2word` is present,
            this is not needed. If both are provided, `dictionary` will be used.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used. 'c_uci' is also known as 'c_pmi' in the literature.
            For 'u_mass', the corpus **MUST** be provided. If `texts` is provided, it will be converted
            to corpus using the dictionary. For 'c_v', 'c_uci' and 'c_npmi', `texts` **MUST** be provided.
            Corpus is not needed.
        window_size : int, optional
            Size of the window to be used for coherence measures using boolean
            sliding window as their probability estimator. For 'u_mass' this doesn't matter.
            If 'None', the default window sizes are used which are:

                * `c_v` - 110
                * `c_uci` - 10
                * `c_npmi` - 10
        topn : int, optional
            Number of top words to be extracted from each topic.
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the coherence value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.corpus = corpus
        self.dictionary = dictionary
        self.coherence = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the coherence score.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            One of the following parameters are expected:

                * `model` - pre-trained topic model of type :class:`~gensim.models.ldamodel.LdaModel`.
                * `topics` - list of tokenized topics.

        Returns
        -------
        float
            The coherence score.

        """
        self.model = None
        self.topics = None
        super(CoherenceMetric, self).set_parameters(**kwargs)
        cm = gensim.models.CoherenceModel(model=self.model, topics=self.topics, texts=self.texts, corpus=self.corpus, dictionary=self.dictionary, window_size=self.window_size, coherence=self.coherence, topn=self.topn)
        return cm.get_coherence()