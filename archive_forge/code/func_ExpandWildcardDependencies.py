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
def ExpandWildcardDependencies(targets, data):
    """Expands dependencies specified as build_file:*.

  For each target in |targets|, examines sections containing links to other
  targets.  If any such section contains a link of the form build_file:*, it
  is taken as a wildcard link, and is expanded to list each target in
  build_file.  The |data| dict provides access to build file dicts.

  Any target that does not wish to be included by wildcard can provide an
  optional "suppress_wildcard" key in its target dict.  When present and
  true, a wildcard dependency link will not include such targets.

  All dependency names, including the keys to |targets| and the values in each
  dependency list, must be qualified when this function is called.
  """
    for target, target_dict in targets.items():
        target_build_file = gyp.common.BuildFile(target)
        for dependency_key in dependency_sections:
            dependencies = target_dict.get(dependency_key, [])
            index = 0
            while index < len(dependencies):
                dependency_build_file, dependency_target, dependency_toolset = gyp.common.ParseQualifiedTarget(dependencies[index])
                if dependency_target != '*' and dependency_toolset != '*':
                    index = index + 1
                    continue
                if dependency_build_file == target_build_file:
                    raise GypError('Found wildcard in ' + dependency_key + ' of ' + target + ' referring to same build file')
                del dependencies[index]
                index = index - 1
                dependency_target_dicts = data[dependency_build_file]['targets']
                for dependency_target_dict in dependency_target_dicts:
                    if int(dependency_target_dict.get('suppress_wildcard', False)):
                        continue
                    dependency_target_name = dependency_target_dict['target_name']
                    if dependency_target != '*' and dependency_target != dependency_target_name:
                        continue
                    dependency_target_toolset = dependency_target_dict['toolset']
                    if dependency_toolset != '*' and dependency_toolset != dependency_target_toolset:
                        continue
                    dependency = gyp.common.QualifiedTarget(dependency_build_file, dependency_target_name, dependency_target_toolset)
                    index = index + 1
                    dependencies.insert(index, dependency)
                index = index + 1