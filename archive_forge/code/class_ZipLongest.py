from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
class ZipLongest(Zip):
    """Iterate over constituent sweeps in parallel

    Analogous to itertools.zip_longest.
    Note that we iterate until all sweeps terminate,
    so if the sweeps are different lengths, the
    shorter sweeps will be filled by repeating their last value
    until all sweeps have equal length.

    Note that this is different from itertools.zip_longest,
    which uses a fixed fill value.

    Raises:
        ValueError if an input sweep if completely empty.
    """

    def __init__(self, *sweeps: Sweep) -> None:
        super().__init__(*sweeps)
        if any((len(sweep) == 0 for sweep in self.sweeps)):
            raise ValueError('All sweeps must be non-empty for ZipLongest')

    def __eq__(self, other):
        if not isinstance(other, ZipLongest):
            return NotImplemented
        return self.sweeps == other.sweeps

    def __len__(self) -> int:
        if not self.sweeps:
            return 0
        return max((len(sweep) for sweep in self.sweeps))

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, tuple(self.sweeps)))

    def __repr__(self) -> str:
        sweeps_repr = ', '.join((repr(s) for s in self.sweeps))
        return f'cirq_google.ZipLongest({sweeps_repr})'

    def __str__(self) -> str:
        sweeps_repr = ', '.join((repr(s) for s in self.sweeps))
        return f'ZipLongest({sweeps_repr})'

    def param_tuples(self) -> Iterator[Params]:

        def _iter_and_repeat_last(one_iter: Iterator[Params]):
            last = None
            for last in one_iter:
                yield last
            while True:
                yield last
        iters = [_iter_and_repeat_last(sweep.param_tuples()) for sweep in self.sweeps]
        for values in itertools.islice(zip(*iters), len(self)):
            yield tuple((item for value in values for item in value))