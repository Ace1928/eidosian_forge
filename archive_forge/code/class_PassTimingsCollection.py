import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
class PassTimingsCollection(Sequence):
    """A collection of pass timings.

    This class implements the ``Sequence`` protocol for accessing the
    individual timing records.
    """

    def __init__(self, name):
        self._name = name
        self._records = []

    @contextmanager
    def record(self, name):
        """Record new timings and append to this collection.

        Note: this is mainly for internal use inside the compiler pipeline.

        See also ``RecordLLVMPassTimings``

        Parameters
        ----------
        name: str
            Name for the records.
        """
        if config.LLVM_PASS_TIMINGS:
            with RecordLLVMPassTimings() as timings:
                yield
            rec = timings.get()
            if rec:
                self._append(name, rec)
        else:
            yield

    def _append(self, name, timings):
        """Append timing records

        Parameters
        ----------
        name: str
            Name for the records.
        timings: ProcessedPassTimings
            the timing records.
        """
        self._records.append(NamedTimings(name, timings))

    def get_total_time(self):
        """Computes the sum of the total time across all contained timings.

        Returns
        -------
        res: float or None
            Returns the total number of seconds or None if no timings were
            recorded
        """
        if self._records:
            return sum((r.timings.get_total_time() for r in self._records))
        else:
            return None

    def list_longest_first(self):
        """Returns the timings in descending order of total time duration.

        Returns
        -------
        res: List[ProcessedPassTimings]
        """
        return sorted(self._records, key=lambda x: x.timings.get_total_time(), reverse=True)

    @property
    def is_empty(self):
        """
        """
        return not self._records

    def summary(self, topn=5):
        """Return a string representing the summary of the timings.

        Parameters
        ----------
        topn: int; optional, default=5.
            This limits the maximum number of items to show.
            This function will show the ``topn`` most time-consuming passes.

        Returns
        -------
        res: str

        See also ``ProcessedPassTimings.summary()``
        """
        if self.is_empty:
            return 'No pass timings were recorded'
        else:
            buf = []
            ap = buf.append
            ap(f'Printing pass timings for {self._name}')
            overall_time = self.get_total_time()
            ap(f'Total time: {overall_time:.4f}')
            for i, r in enumerate(self._records):
                ap(f'== #{i} {r.name}')
                percent = r.timings.get_total_time() / overall_time * 100
                ap(f' Percent: {percent:.1f}%')
                ap(r.timings.summary(topn=topn, indent=1))
            return '\n'.join(buf)

    def __getitem__(self, i):
        """Get the i-th timing record.

        Returns
        -------
        res: (name, timings)
            A named tuple with two fields:

            - name: str
            - timings: ProcessedPassTimings
        """
        return self._records[i]

    def __len__(self):
        """Length of this collection.
        """
        return len(self._records)

    def __str__(self):
        return self.summary()