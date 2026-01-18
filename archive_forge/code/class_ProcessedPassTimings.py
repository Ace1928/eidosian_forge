import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
class ProcessedPassTimings:
    """A class for processing raw timing report from LLVM.

    The processing is done lazily so we don't waste time processing unused
    timing information.
    """

    def __init__(self, raw_data):
        self._raw_data = raw_data

    def __bool__(self):
        return bool(self._raw_data)

    def get_raw_data(self):
        """Returns the raw string data.

        Returns
        -------
        res: str
        """
        return self._raw_data

    def get_total_time(self):
        """Compute the total time spend in all passes.

        Returns
        -------
        res: float
        """
        return self.list_records()[-1].wall_time

    def list_records(self):
        """Get the processed data for the timing report.

        Returns
        -------
        res: List[PassTimingRecord]
        """
        return self._processed

    def list_top(self, n):
        """Returns the top(n) most time-consuming (by wall-time) passes.

        Parameters
        ----------
        n: int
            This limits the maximum number of items to show.
            This function will show the ``n`` most time-consuming passes.

        Returns
        -------
        res: List[PassTimingRecord]
            Returns the top(n) most time-consuming passes in descending order.
        """
        records = self.list_records()
        key = operator.attrgetter('wall_time')
        return heapq.nlargest(n, records[:-1], key)

    def summary(self, topn=5, indent=0):
        """Return a string summarizing the timing information.

        Parameters
        ----------
        topn: int; optional
            This limits the maximum number of items to show.
            This function will show the ``topn`` most time-consuming passes.
        indent: int; optional
            Set the indentation level. Defaults to 0 for no indentation.

        Returns
        -------
        res: str
        """
        buf = []
        prefix = ' ' * indent

        def ap(arg):
            buf.append(f'{prefix}{arg}')
        ap(f'Total {self.get_total_time():.4f}s')
        ap('Top timings:')
        for p in self.list_top(topn):
            ap(f'  {p.wall_time:.4f}s ({p.wall_percent:5}%) {p.pass_name}')
        return '\n'.join(buf)

    @cached_property
    def _processed(self):
        """A cached property for lazily processing the data and returning it.

        See ``_process()`` for details.
        """
        return self._process()

    def _process(self):
        """Parses the raw string data from LLVM timing report and attempts
        to improve the data by recomputing the times
        (See `_adjust_timings()``).
        """

        def parse(raw_data):
            """A generator that parses the raw_data line-by-line to extract
            timing information for each pass.
            """
            lines = raw_data.splitlines()
            colheader = '[a-zA-Z+ ]+'
            multicolheaders = f'(?:\\s*-+{colheader}-+)+'
            line_iter = iter(lines)
            header_map = {'User Time': 'user', 'System Time': 'system', 'User+System': 'user_system', 'Wall Time': 'wall', 'Instr': 'instruction', 'Name': 'pass_name'}
            for ln in line_iter:
                m = re.match(multicolheaders, ln)
                if m:
                    raw_headers = re.findall('[a-zA-Z][a-zA-Z+ ]+', ln)
                    headers = [header_map[k.strip()] for k in raw_headers]
                    break
            assert headers[-1] == 'pass_name'
            attrs = []
            n = '\\s*((?:[0-9]+\\.)?[0-9]+)'
            pat = ''
            for k in headers[:-1]:
                if k == 'instruction':
                    pat += n
                else:
                    attrs.append(f'{k}_time')
                    attrs.append(f'{k}_percent')
                    pat += f'\\s+(?:{n}\\s*\\({n}%\\)|-+)'
            missing = {}
            for k in PassTimingRecord._fields:
                if k not in attrs and k != 'pass_name':
                    missing[k] = 0.0
            pat += '\\s*(.*)'
            for ln in line_iter:
                m = re.match(pat, ln)
                if m is not None:
                    raw_data = list(m.groups())
                    data = {k: float(v) if v is not None else 0.0 for k, v in zip(attrs, raw_data)}
                    data.update(missing)
                    pass_name = raw_data[-1]
                    rec = PassTimingRecord(pass_name=pass_name, **data)
                    yield rec
                    if rec.pass_name == 'Total':
                        break
            remaining = '\n'.join(line_iter)
            if remaining:
                raise ValueError(f'unexpected text after parser finished:\n{remaining}')
        records = list(parse(self._raw_data))
        return _adjust_timings(records)