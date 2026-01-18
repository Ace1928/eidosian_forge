from __future__ import unicode_literals
from .dag import get_outgoing_edges, topo_sort
from ._utils import basestring, convert_kwargs_to_cmd_line_args
from builtins import str
from functools import reduce
import collections
import copy
import operator
import subprocess
from ._ffmpeg import input, output
from .nodes import (
def _get_global_args(node):
    return list(node.args)