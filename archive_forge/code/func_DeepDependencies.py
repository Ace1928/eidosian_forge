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
def DeepDependencies(self, dependencies=None):
    """Returns an OrderedSet of all of a target's dependencies, recursively."""
    if dependencies is None:
        dependencies = OrderedSet()
    for dependency in self.dependencies:
        if dependency.ref is None:
            continue
        if dependency.ref not in dependencies:
            dependency.DeepDependencies(dependencies)
            dependencies.add(dependency.ref)
    return dependencies