import importlib
import math
import re
from enum import Enum
class DictFrequencies:
    """
    Dict freqs.
    """

    def __init__(self, freqs):
        self.freqs = freqs
        self.N = sum(freqs.values())
        self.V = len(freqs)
        self.logNV = math.log(self.N + self.V)