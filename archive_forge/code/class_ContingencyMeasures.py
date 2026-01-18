import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
class ContingencyMeasures:
    """Wraps NgramAssocMeasures classes such that the arguments of association
    measures are contingency table values rather than marginals.
    """

    def __init__(self, measures):
        """Constructs a ContingencyMeasures given a NgramAssocMeasures class"""
        self.__class__.__name__ = 'Contingency' + measures.__class__.__name__
        for k in dir(measures):
            if k.startswith('__'):
                continue
            v = getattr(measures, k)
            if not k.startswith('_'):
                v = self._make_contingency_fn(measures, v)
            setattr(self, k, v)

    @staticmethod
    def _make_contingency_fn(measures, old_fn):
        """From an association measure function, produces a new function which
        accepts contingency table values as its arguments.
        """

        def res(*contingency):
            return old_fn(*measures._marginals(*contingency))
        res.__doc__ = old_fn.__doc__
        res.__name__ = old_fn.__name__
        return res