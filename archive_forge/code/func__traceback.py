import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
def _traceback(self, job):
    num = job if isinstance(job, int) else job.num
    try:
        self.all[num].traceback()
    except KeyError:
        error('Job #%s not found' % num)