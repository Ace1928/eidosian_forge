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
def _reset_hierarchy(self):
    """Reset the hierarchy on a graph"""
    for node in self._graph.nodes():
        if isinstance(node, Workflow):
            node._reset_hierarchy()
            for innernode in node._graph.nodes():
                innernode._hierarchy = '.'.join((self.name, innernode._hierarchy))
        else:
            node._hierarchy = self.name