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
def _get_input_args(input_node):
    if input_node.name == input.__name__:
        kwargs = copy.copy(input_node.kwargs)
        filename = kwargs.pop('filename')
        fmt = kwargs.pop('format', None)
        video_size = kwargs.pop('video_size', None)
        args = []
        if fmt:
            args += ['-f', fmt]
        if video_size:
            args += ['-video_size', '{}x{}'.format(video_size[0], video_size[1])]
        args += convert_kwargs_to_cmd_line_args(kwargs)
        args += ['-i', filename]
    else:
        raise ValueError('Unsupported input node: {}'.format(input_node))
    return args