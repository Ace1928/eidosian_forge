import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _convert_to_binary_features(self, features):
    """
        :param features: list of feature string which is needed to convert to binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is 'featureID:value' pairs
        """
    unsorted_result = []
    for feature in features:
        self._dictionary.setdefault(feature, len(self._dictionary))
        unsorted_result.append(self._dictionary[feature])
    return ' '.join((str(featureID) + ':1.0' for featureID in sorted(unsorted_result)))