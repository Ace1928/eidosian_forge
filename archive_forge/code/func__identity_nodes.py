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
def _identity_nodes(graph, include_iterables):
    """Return the IdentityInterface nodes in the graph

    The nodes are in topological sort order. The iterable nodes
    are included if and only if the include_iterables flag is set
    to True.
    """
    import networkx as nx
    return [node for node in nx.topological_sort(graph) if isinstance(node.interface, IdentityInterface) and (include_iterables or getattr(node, 'iterables') is None)]