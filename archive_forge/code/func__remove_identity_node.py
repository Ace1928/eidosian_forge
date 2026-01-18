import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _remove_identity_node(graph, node):
    """Remove identity nodes from an execution graph"""
    portinputs, portoutputs = _node_ports(graph, node)
    for field, connections in list(portoutputs.items()):
        if portinputs:
            _propagate_internal_output(graph, node, field, connections, portinputs)
        else:
            _propagate_root_output(graph, node, field, connections)
    graph.remove_nodes_from([node])
    logger.debug('Removed the identity node %s from the graph.', node)