import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def FlattenToList(self):
    flat_list = OrderedSet()

    def ExtractNodeRef(node):
        """Extracts the object that the node represents from the given node."""
        return node.ref
    in_degree_zeros = sorted(self.dependents[:], key=ExtractNodeRef)
    while in_degree_zeros:
        node = in_degree_zeros.pop()
        flat_list.add(node.ref)
        for node_dependent in sorted(node.dependents, key=ExtractNodeRef):
            is_in_degree_zero = True
            for node_dependent_dependency in sorted(node_dependent.dependencies, key=ExtractNodeRef):
                if node_dependent_dependency.ref not in flat_list:
                    is_in_degree_zero = False
                    break
            if is_in_degree_zero:
                in_degree_zeros += [node_dependent]
    return list(flat_list)