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
def RemoveDuplicateDependencies(targets):
    """Makes sure every dependency appears only once in all targets's dependency
  lists."""
    for target_name, target_dict in targets.items():
        for dependency_key in dependency_sections:
            dependencies = target_dict.get(dependency_key, [])
            if dependencies:
                target_dict[dependency_key] = Unify(dependencies)