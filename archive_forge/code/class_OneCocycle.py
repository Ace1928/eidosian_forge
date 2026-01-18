from ... import sage_helper
from .. import t3mlite as t3m
class OneCocycle:
    """
    A cocycle on the 1-skeleton of a DualCellulation.
    """

    def __init__(self, cellulation, weights):
        self.cellulation, self.weights = (cellulation, weights)
        assert sorted((edge.index for edge in cellulation.edges)) == list(range(len(weights)))
        assert cellulation.B2().transpose() * vector(weights) == 0

    def __call__(self, other):
        if isinstance(other, OneCycle):
            return sum((a * b for a, b in zip(self.weights, other.weights)))