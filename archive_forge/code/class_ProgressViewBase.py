from __future__ import absolute_import, print_function, division
import abc
import logging
import sys
import time
from petl.compat import PY3
from petl.util.base import Table
from petl.util.statistics import onlinestats
class ProgressViewBase(Table):
    """
    Abstract base class for reporting on proecessing status
    """

    def __init__(self, inner, batchsize, prefix):
        self.inner = inner
        self.batchsize = batchsize
        self.prefix = prefix

    @abc.abstractmethod
    def print_message(self, message):
        pass

    def __iter__(self):
        start = time.time()
        batchstart = start
        batchn = 0
        batchtimemin, batchtimemax = (None, None)
        batchtimemean, batchtimevar = (0, 0)
        batchratemean, batchratevar = (0, 0)
        for n, r in enumerate(self.inner):
            if n % self.batchsize == 0 and n > 0:
                batchn += 1
                batchend = time.time()
                batchtime = batchend - batchstart
                if batchtimemin is None or batchtime < batchtimemin:
                    batchtimemin = batchtime
                if batchtimemax is None or batchtime > batchtimemax:
                    batchtimemax = batchtime
                elapsedtime = batchend - start
                try:
                    rate = int(n / elapsedtime)
                except ZeroDivisionError:
                    rate = 0
                try:
                    batchrate = int(self.batchsize / batchtime)
                except ZeroDivisionError:
                    batchrate = 0
                v = (n, elapsedtime, rate, batchtime, batchrate)
                message = self.prefix + '%s rows in %.2fs (%s row/s); batch in %.2fs (%s row/s)' % v
                self.print_message(message)
                batchstart = batchend
                batchtimemean, batchtimevar = onlinestats(batchtime, batchn, mean=batchtimemean, variance=batchtimevar)
                batchratemean, batchratevar = onlinestats(batchrate, batchn, mean=batchratemean, variance=batchratevar)
            yield r
        end = time.time()
        elapsedtime = end - start
        try:
            rate = int(n / elapsedtime)
        except ZeroDivisionError:
            rate = 0
        if batchn > 1:
            if batchtimemin is None:
                batchtimemin = 0
            if batchtimemax is None:
                batchtimemax = 0
            try:
                batchratemin = int(self.batchsize / batchtimemax)
            except ZeroDivisionError:
                batchratemin = 0
            try:
                batchratemax = int(self.batchsize / batchtimemin)
            except ZeroDivisionError:
                batchratemax = 0
            v = (n, elapsedtime, rate, batchtimemean, batchtimevar ** 0.5, batchtimemin, batchtimemax, int(batchratemean), int(batchratevar ** 0.5), int(batchratemin), int(batchratemax))
            message = self.prefix + '%s rows in %.2fs (%s row/s); batches in %.2f +/- %.2fs [%.2f-%.2f] (%s +/- %s rows/s [%s-%s])' % v
        else:
            v = (n, elapsedtime, rate)
            message = self.prefix + '%s rows in %.2fs (%s row/s)' % v
        self.print_message(message)