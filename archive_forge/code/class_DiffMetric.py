import gensim
import logging
import copy
import sys
import numpy as np
class DiffMetric(Metric):
    """Metric class for topic difference evaluation."""

    def __init__(self, distance='jaccard', num_words=100, n_ann_terms=10, diagonal=True, annotation=False, normed=True, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        distance : {'kullback_leibler', 'hellinger', 'jaccard'}, optional
            Measure used to calculate difference between any topic pair.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the coherence value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the difference between each pair of topics in two topic models.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            Two models of type :class:`~gensim.models.ldamodelLdaModel`
            are expected using the keys `model` and `other_model`.

        Returns
        -------
        np.ndarray of shape (`model.num_topics`, `other_model.num_topics`)
            Matrix of differences between each pair of topics.
        np.ndarray of shape (`model.num_topics`, `other_model.num_topics`, 2), optional
            Annotation matrix where for each pair we include the word from the intersection of the two topics,
            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.

        """
        super(DiffMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(self.other_model, self.distance, self.num_words, self.n_ann_terms, self.diagonal, self.annotation, self.normed)
        return diff_diagonal