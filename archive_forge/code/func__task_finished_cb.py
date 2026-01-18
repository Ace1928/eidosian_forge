import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
def _task_finished_cb(self, jobid, cached=False):
    """Extract outputs and assign to inputs of dependent tasks

        This is called when a job is completed.
        """
    logger.info('[Job %d] %s (%s).', jobid, 'Cached' if cached else 'Completed', self.procs[jobid])
    if self._status_callback:
        self._status_callback(self.procs[jobid], 'end')
    self.proc_pending[jobid] = False
    rowview = self.depidx.getrowview(jobid)
    rowview[rowview.nonzero()] = 0
    if jobid not in self.mapnodesubids:
        self.refidx[self.refidx[:, jobid].nonzero()[0], jobid] = 0