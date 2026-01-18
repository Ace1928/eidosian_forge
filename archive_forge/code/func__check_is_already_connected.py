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
def _check_is_already_connected(workflow, node, attrname):
    for _, _, d in workflow._graph.in_edges(nbunch=node, data=True):
        for cd in d['connect']:
            if attrname == cd[1]:
                return False
    return True