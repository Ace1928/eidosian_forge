import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
@staticmethod
def _stringify_node(node):
    if 'target' in node.metadata:
        return '%s (link to %s)' % (node.item, node.metadata['target'])
    else:
        return str(node.item)