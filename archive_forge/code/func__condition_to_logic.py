from abc import ABCMeta, abstractmethod
from nltk import jsontags
def _condition_to_logic(feature, value):
    """
            Return a compact, predicate-logic styled string representation
            of the given condition.
            """
    return '{}:{}@[{}]'.format(feature.PROPERTY_NAME, value, ','.join((str(w) for w in feature.positions)))