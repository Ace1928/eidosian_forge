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
def _has_attr(self, parameter, subtype='in'):
    """Checks if a parameter is available as an input or output"""
    hierarchy = parameter.split('.')
    if len(hierarchy) < 2:
        return False
    attrname = hierarchy.pop()
    nodename = hierarchy.pop()

    def _check_is_already_connected(workflow, node, attrname):
        for _, _, d in workflow._graph.in_edges(nbunch=node, data=True):
            for cd in d['connect']:
                if attrname == cd[1]:
                    return False
        return True
    targetworkflow = self
    while hierarchy:
        workflowname = hierarchy.pop(0)
        workflow = None
        for node in targetworkflow._graph.nodes():
            if node.name == workflowname:
                if isinstance(node, Workflow):
                    workflow = node
                    break
        if workflow is None:
            return False
        if subtype == 'in':
            hierattrname = '.'.join(hierarchy + [nodename, attrname])
            if not _check_is_already_connected(targetworkflow, workflow, hierattrname):
                return False
        targetworkflow = workflow
    targetnode = None
    for node in targetworkflow._graph.nodes():
        if node.name == nodename:
            if isinstance(node, Workflow):
                return False
            else:
                targetnode = node
                break
    if targetnode is None:
        return False
    if subtype == 'in':
        if not hasattr(targetnode.inputs, attrname):
            return False
    elif not hasattr(targetnode.outputs, attrname):
        return False
    if subtype == 'in':
        if not _check_is_already_connected(targetworkflow, targetnode, attrname):
            return False
    return True