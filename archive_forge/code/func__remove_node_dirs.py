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
def _remove_node_dirs(self):
    """Removes directories whose outputs have already been used up"""
    if str2bool(self._config['execution']['remove_node_directories']):
        indices = np.nonzero((self.refidx.sum(axis=1) == 0).__array__())[0]
        for idx in indices:
            if idx in self.mapnodesubids:
                continue
            if self.proc_done[idx] and (not self.proc_pending[idx]):
                self.refidx[idx, idx] = -1
                outdir = self.procs[idx].output_dir()
                logger.info('[node dependencies finished] removing node: %s from directory %s' % (self.procs[idx]._id, outdir))
                shutil.rmtree(outdir)