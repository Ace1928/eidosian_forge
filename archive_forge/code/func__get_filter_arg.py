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
def _get_filter_arg(filter_nodes, outgoing_edge_maps, stream_name_map):
    _allocate_filter_stream_names(filter_nodes, outgoing_edge_maps, stream_name_map)
    filter_specs = [_get_filter_spec(node, outgoing_edge_maps[node], stream_name_map) for node in filter_nodes]
    return ';'.join(filter_specs)