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
def _generate_dependency_list(self, graph):
    """Generates a dependency list for a list of graphs."""
    self.procs, _ = topological_sort(graph)
    self.depidx = _graph_to_lil_matrix(graph, nodelist=self.procs)
    self.refidx = self.depidx.astype(int)
    self.proc_done = np.zeros(len(self.procs), dtype=bool)
    self.proc_pending = np.zeros(len(self.procs), dtype=bool)