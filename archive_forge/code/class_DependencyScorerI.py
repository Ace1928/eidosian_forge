import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
class DependencyScorerI:
    """
    A scorer for calculated the weights on the edges of a weighted
    dependency graph.  This is used by a
    ``ProbabilisticNonprojectiveParser`` to initialize the edge
    weights of a ``DependencyGraph``.  While typically this would be done
    by training a binary classifier, any class that can return a
    multidimensional list representation of the edge weights can
    implement this interface.  As such, it has no necessary
    fields.
    """

    def __init__(self):
        if self.__class__ == DependencyScorerI:
            raise TypeError('DependencyScorerI is an abstract interface')

    def train(self, graphs):
        """
        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
            Typically the edges present in the graphs can be used as
            positive training examples, and the edges not present as negative
            examples.
        """
        raise NotImplementedError()

    def score(self, graph):
        """
        :type graph: DependencyGraph
        :param graph: A dependency graph whose set of edges need to be
            scored.
        :rtype: A three-dimensional list of numbers.
        :return: The score is returned in a multidimensional(3) list, such
            that the outer-dimension refers to the head, and the
            inner-dimension refers to the dependencies.  For instance,
            scores[0][1] would reference the list of scores corresponding to
            arcs from node 0 to node 1.  The node's 'address' field can be used
            to determine its number identification.

        For further illustration, a score list corresponding to Fig.2 of
        Keith Hall's 'K-best Spanning Tree Parsing' paper::

              scores = [[[], [5],  [1],  [1]],
                       [[], [],   [11], [4]],
                       [[], [10], [],   [5]],
                       [[], [8],  [8],  []]]

        When used in conjunction with a MaxEntClassifier, each score would
        correspond to the confidence of a particular edge being classified
        with the positive training examples.
        """
        raise NotImplementedError()