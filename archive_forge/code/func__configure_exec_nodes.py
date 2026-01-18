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
def _configure_exec_nodes(self, graph):
    """Ensure that each node knows where to get inputs from"""
    for node in graph.nodes():
        node.input_source = {}
        for edge in graph.in_edges(node):
            data = graph.get_edge_data(*edge)
            for sourceinfo, field in data['connect']:
                node.input_source[field] = (op.join(edge[0].output_dir(), 'result_%s.pklz' % edge[0].name), sourceinfo)