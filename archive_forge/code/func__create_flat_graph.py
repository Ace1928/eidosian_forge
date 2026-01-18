import os
import os.path as op
import sys
from datetime import datetime
from copy import deepcopy
import pickle
import shutil
import numpy as np
from ... import config, logging
from ...utils.misc import str2bool
from ...utils.functions import getsource, create_function_from_source
from ...interfaces.base import traits, TraitedSpec, TraitDictObject, TraitListObject
from ...utils.filemanip import save_json
from .utils import (
from .base import EngineBase
from .nodes import MapNode
def _create_flat_graph(self):
    """Make a simple DAG where no node is a workflow."""
    logger.debug('Creating flat graph for workflow: %s', self.name)
    workflowcopy = deepcopy(self)
    workflowcopy._generate_flatgraph()
    return workflowcopy._graph