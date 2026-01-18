import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
class NaiveBayesDependencyScorer(DependencyScorerI):
    """
    A dependency scorer built around a MaxEnt classifier.  In this
    particular class that classifier is a ``NaiveBayesClassifier``.
    It uses head-word, head-tag, child-word, and child-tag features
    for classification.

    >>> from nltk.parse.dependencygraph import DependencyGraph, conll_data2

    >>> graphs = [DependencyGraph(entry) for entry in conll_data2.split('\\n\\n') if entry]
    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train(graphs, NaiveBayesDependencyScorer())
    >>> parses = npp.parse(['Cathy', 'zag', 'hen', 'zwaaien', '.'], ['N', 'V', 'Pron', 'Adj', 'N', 'Punc'])
    >>> len(list(parses))
    1

    """

    def __init__(self):
        pass

    def train(self, graphs):
        """
        Trains a ``NaiveBayesClassifier`` using the edges present in
        graphs list as positive examples, the edges not present as
        negative examples.  Uses a feature vector of head-word,
        head-tag, child-word, and child-tag.

        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
        """
        from nltk.classify import NaiveBayesClassifier
        labeled_examples = []
        for graph in graphs:
            for head_node in graph.nodes.values():
                for child_index, child_node in graph.nodes.items():
                    if child_index in head_node['deps']:
                        label = 'T'
                    else:
                        label = 'F'
                    labeled_examples.append((dict(a=head_node['word'], b=head_node['tag'], c=child_node['word'], d=child_node['tag']), label))
        self.classifier = NaiveBayesClassifier.train(labeled_examples)

    def score(self, graph):
        """
        Converts the graph into a feature-based representation of
        each edge, and then assigns a score to each based on the
        confidence of the classifier in assigning it to the
        positive label.  Scores are returned in a multidimensional list.

        :type graph: DependencyGraph
        :param graph: A dependency graph to score.
        :rtype: 3 dimensional list
        :return: Edge scores for the graph parameter.
        """
        edges = []
        for head_node in graph.nodes.values():
            for child_node in graph.nodes.values():
                edges.append(dict(a=head_node['word'], b=head_node['tag'], c=child_node['word'], d=child_node['tag']))
        edge_scores = []
        row = []
        count = 0
        for pdist in self.classifier.prob_classify_many(edges):
            logger.debug('%.4f %.4f', pdist.prob('T'), pdist.prob('F'))
            row.append([math.log(pdist.prob('T') + 1e-11)])
            count += 1
            if count == len(graph.nodes):
                edge_scores.append(row)
                row = []
                count = 0
        return edge_scores