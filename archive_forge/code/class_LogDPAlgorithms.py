import warnings
from Bio import BiopythonDeprecationWarning
class LogDPAlgorithms(AbstractDPAlgorithms):
    """Implement forward and backward algorithms using a log approach.

    This uses the approach of calculating the sum of log probabilities
    using a lookup table for common values.

    XXX This is not implemented yet!
    """

    def __init__(self, markov_model, sequence):
        """Initialize the class."""
        raise NotImplementedError("Haven't coded this yet...")