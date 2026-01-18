from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
@property
def _gate_names(self):
    return ['_r', '_z', '_o']