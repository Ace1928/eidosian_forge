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
def _clean_queue(self, jobid, graph, result=None):
    logger.debug('Clearing %d from queue', jobid)
    if self._status_callback:
        self._status_callback(self.procs[jobid], 'exception')
    if result is None:
        result = {'result': None, 'traceback': '\n'.join(format_exception(*sys.exc_info()))}
    crashfile = self._report_crash(self.procs[jobid], result=result)
    if str2bool(self._config['execution']['stop_on_first_crash']):
        raise RuntimeError(''.join(result['traceback']))
    if jobid in self.mapnodesubids:
        self.proc_pending[jobid] = False
        self.proc_done[jobid] = True
        jobid = self.mapnodesubids[jobid]
        self.proc_pending[jobid] = False
        self.proc_done[jobid] = True
    return self._remove_node_deps(jobid, crashfile, graph)