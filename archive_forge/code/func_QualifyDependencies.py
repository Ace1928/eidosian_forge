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
def QualifyDependencies(targets):
    """Make dependency links fully-qualified relative to the current directory.

  |targets| is a dict mapping fully-qualified target names to their target
  dicts.  For each target in this dict, keys known to contain dependency
  links are examined, and any dependencies referenced will be rewritten
  so that they are fully-qualified and relative to the current directory.
  All rewritten dependencies are suitable for use as keys to |targets| or a
  similar dict.
  """
    all_dependency_sections = [dep + op for dep in dependency_sections for op in ('', '!', '/')]
    for target, target_dict in targets.items():
        target_build_file = gyp.common.BuildFile(target)
        toolset = target_dict['toolset']
        for dependency_key in all_dependency_sections:
            dependencies = target_dict.get(dependency_key, [])
            for index, dep in enumerate(dependencies):
                dep_file, dep_target, dep_toolset = gyp.common.ResolveTarget(target_build_file, dep, toolset)
                if not multiple_toolsets:
                    dep_toolset = toolset
                dependency = gyp.common.QualifiedTarget(dep_file, dep_target, dep_toolset)
                dependencies[index] = dependency
                if dependency_key != 'dependencies' and dependency not in target_dict['dependencies']:
                    raise GypError('Found ' + dependency + ' in ' + dependency_key + ' of ' + target + ', but not in dependencies')