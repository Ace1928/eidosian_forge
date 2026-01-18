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
def _check_nodes(self, nodes):
    """Checks if any of the nodes are already in the graph"""
    node_names = [node.name for node in self._graph.nodes()]
    node_lineage = [node._hierarchy for node in self._graph.nodes()]
    for node in nodes:
        if node.name in node_names:
            idx = node_names.index(node.name)
            try:
                this_node_lineage = node_lineage[idx]
            except IndexError:
                raise IOError('Duplicate node name "%s" found.' % node.name)
            else:
                if this_node_lineage in [node._hierarchy, self.name]:
                    raise IOError('Duplicate node name "%s" found.' % node.name)
        else:
            node_names.append(node.name)