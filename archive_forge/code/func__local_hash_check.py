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
def _local_hash_check(self, jobid, graph):
    if not str2bool(self.procs[jobid].config['execution']['local_hash_check']):
        return False
    try:
        cached, updated = self.procs[jobid].is_cached()
    except Exception:
        logger.warning('Error while checking node hash, forcing re-run. Although this error may not prevent the workflow from running, it could indicate a major problem. Please report a new issue at https://github.com/nipy/nipype/issues adding the following information:\n\n\tNode: %s\n\tInterface: %s.%s\n\tTraceback:\n%s', self.procs[jobid], self.procs[jobid].interface.__module__, self.procs[jobid].interface.__class__.__name__, '\n'.join(format_exception(*sys.exc_info())))
        return False
    logger.debug('Checking hash "%s" locally: cached=%s, updated=%s.', self.procs[jobid], cached, updated)
    overwrite = self.procs[jobid].overwrite
    always_run = self.procs[jobid].interface.always_run
    if cached and updated and (overwrite is False or (overwrite is None and (not always_run))):
        logger.debug('Skipping cached node %s with ID %s.', self.procs[jobid], jobid)
        try:
            self._task_finished_cb(jobid, cached=True)
            self._remove_node_dirs()
        except Exception:
            logger.debug('Error skipping cached node %s (%s).\n\n%s', self.procs[jobid], jobid, '\n'.join(format_exception(*sys.exc_info())))
            self._clean_queue(jobid, graph)
            self.proc_pending[jobid] = False
        return True
    return False