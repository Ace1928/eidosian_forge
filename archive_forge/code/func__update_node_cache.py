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
def _update_node_cache(self):
    nodes = set(self._graph)
    added_nodes = nodes.difference(self._nodes_cache)
    removed_nodes = self._nodes_cache.difference(nodes)
    self._nodes_cache = nodes
    self._nested_workflows_cache.difference_update(removed_nodes)
    for node in added_nodes:
        if isinstance(node, Workflow):
            self._nested_workflows_cache.add(node)