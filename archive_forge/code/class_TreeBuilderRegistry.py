from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
class TreeBuilderRegistry(object):
    """A way of looking up TreeBuilder subclasses by their name or by desired
    features.
    """

    def __init__(self):
        self.builders_for_feature = defaultdict(list)
        self.builders = []

    def register(self, treebuilder_class):
        """Register a treebuilder based on its advertised features.

        :param treebuilder_class: A subclass of Treebuilder. its .features
           attribute should list its features.
        """
        for feature in treebuilder_class.features:
            self.builders_for_feature[feature].insert(0, treebuilder_class)
        self.builders.insert(0, treebuilder_class)

    def lookup(self, *features):
        """Look up a TreeBuilder subclass with the desired features.

        :param features: A list of features to look for. If none are
            provided, the most recently registered TreeBuilder subclass
            will be used.
        :return: A TreeBuilder subclass, or None if there's no
            registered subclass with all the requested features.
        """
        if len(self.builders) == 0:
            return None
        if len(features) == 0:
            return self.builders[0]
        features = list(features)
        features.reverse()
        candidates = None
        candidate_set = None
        while len(features) > 0:
            feature = features.pop()
            we_have_the_feature = self.builders_for_feature.get(feature, [])
            if len(we_have_the_feature) > 0:
                if candidates is None:
                    candidates = we_have_the_feature
                    candidate_set = set(candidates)
                else:
                    candidate_set = candidate_set.intersection(set(we_have_the_feature))
        if candidate_set is None:
            return None
        for candidate in candidates:
            if candidate in candidate_set:
                return candidate
        return None