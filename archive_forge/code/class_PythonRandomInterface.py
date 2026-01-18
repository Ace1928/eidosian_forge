import sys
import uuid
import warnings
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Sized
from itertools import chain, tee
import networkx as nx
class PythonRandomInterface:

    def __init__(self, rng=None):
        try:
            import numpy as np
        except ImportError:
            msg = 'numpy not found, only random.random available.'
            warnings.warn(msg, ImportWarning)
        if rng is None:
            self._rng = np.random.mtrand._rand
        else:
            self._rng = rng

    def random(self):
        return self._rng.random()

    def uniform(self, a, b):
        return a + (b - a) * self._rng.random()

    def randrange(self, a, b=None):
        import numpy as np
        if isinstance(self._rng, np.random.Generator):
            return self._rng.integers(a, b)
        return self._rng.randint(a, b)

    def choice(self, seq):
        import numpy as np
        if isinstance(self._rng, np.random.Generator):
            idx = self._rng.integers(0, len(seq))
        else:
            idx = self._rng.randint(0, len(seq))
        return seq[idx]

    def gauss(self, mu, sigma):
        return self._rng.normal(mu, sigma)

    def shuffle(self, seq):
        return self._rng.shuffle(seq)

    def sample(self, seq, k):
        return self._rng.choice(list(seq), size=(k,), replace=False)

    def randint(self, a, b):
        import numpy as np
        if isinstance(self._rng, np.random.Generator):
            return self._rng.integers(a, b + 1)
        return self._rng.randint(a, b + 1)

    def expovariate(self, scale):
        return self._rng.exponential(1 / scale)

    def paretovariate(self, shape):
        return self._rng.pareto(shape)