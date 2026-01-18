import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
@staticmethod
def _metadata_path_selector(root_node, child_node):
    return child_node.metadata['path']