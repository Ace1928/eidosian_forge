import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
def _up_to_root_selector(self, root_node, child_node):
    path_pieces = [child_node.item]
    for parent_node in child_node.path_iter(include_self=False):
        if parent_node is root_node:
            break
        path_pieces.append(parent_node.item)
    if len(path_pieces) > 1:
        path_pieces.reverse()
    return self.join(*path_pieces)