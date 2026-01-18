import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
class DemoScorer(DependencyScorerI):

    def train(self, graphs):
        print('Training...')

    def score(self, graph):
        return [[[], [5], [1], [1]], [[], [], [11], [4]], [[], [10], [], [5]], [[], [8], [8], []]]