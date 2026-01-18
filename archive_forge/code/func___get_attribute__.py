import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def __get_attribute__(self, attr):
    """Look for default attributes for this node"""
    attr_val = self.obj_dict['attributes'].get(attr, None)
    if attr_val is None:
        default_node_name = self.obj_dict['type']
        if default_node_name in ('subgraph', 'digraph', 'cluster'):
            default_node_name = 'graph'
        g = self.get_parent_graph()
        if g is not None:
            defaults = g.get_node(default_node_name)
        else:
            return None
        if not isinstance(defaults, (list, tuple)):
            defaults = [defaults]
        for default in defaults:
            attr_val = default.obj_dict['attributes'].get(attr, None)
            if attr_val:
                return attr_val
    else:
        return attr_val
    return None