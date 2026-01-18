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
def FindCycles(self):
    """
    Returns a list of cycles in the graph, where each cycle is its own list.
    """
    results = []
    visited = set()

    def Visit(node, path):
        for child in node.dependents:
            if child in path:
                results.append([child] + path[:path.index(child) + 1])
            elif child not in visited:
                visited.add(child)
                Visit(child, [child] + path)
    visited.add(self)
    Visit(self, [self])
    return results