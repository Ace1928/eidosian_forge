import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
def _group_flush(self, group, name):
    """Flush a given job group

        Return True if the group had any elements."""
    njobs = len(group)
    if njobs:
        plural = {1: ''}.setdefault(njobs, 's')
        print('Flushing %s %s job%s.' % (njobs, name, plural))
        group[:] = []
        return True