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
def _set_needed_outputs(self, graph):
    """Initialize node with list of which outputs are needed."""
    rm_outputs = self.config['execution']['remove_unnecessary_outputs']
    if not str2bool(rm_outputs):
        return
    for node in graph.nodes():
        node.needed_outputs = []
        for edge in graph.out_edges(node):
            data = graph.get_edge_data(*edge)
            sourceinfo = [v1[0] if isinstance(v1, tuple) else v1 for v1, v2 in data['connect']]
            node.needed_outputs += [v for v in sourceinfo if v not in node.needed_outputs]
        if node.needed_outputs:
            node.needed_outputs = sorted(node.needed_outputs)