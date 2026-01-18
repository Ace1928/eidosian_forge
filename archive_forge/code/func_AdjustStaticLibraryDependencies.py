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
def AdjustStaticLibraryDependencies(flat_list, targets, dependency_nodes, sort_dependencies):
    for target in flat_list:
        target_dict = targets[target]
        target_type = target_dict['type']
        if target_type == 'static_library':
            if 'dependencies' not in target_dict:
                continue
            target_dict['dependencies_original'] = target_dict.get('dependencies', [])[:]
            dependencies = dependency_nodes[target].DirectAndImportedDependencies(targets)
            index = 0
            while index < len(dependencies):
                dependency = dependencies[index]
                dependency_dict = targets[dependency]
                if dependency_dict['type'] == 'static_library' and (not dependency_dict.get('hard_dependency', False)) or (dependency_dict['type'] != 'static_library' and dependency not in target_dict['dependencies']):
                    del dependencies[index]
                else:
                    index = index + 1
            if len(dependencies) > 0:
                target_dict['dependencies'] = dependencies
            else:
                del target_dict['dependencies']
        elif target_type in linkable_types:
            link_dependencies = dependency_nodes[target].DependenciesToLinkAgainst(targets)
            for dependency in link_dependencies:
                if dependency == target:
                    continue
                if 'dependencies' not in target_dict:
                    target_dict['dependencies'] = []
                if dependency not in target_dict['dependencies']:
                    target_dict['dependencies'].append(dependency)
            if sort_dependencies and 'dependencies' in target_dict:
                target_dict['dependencies'] = [dep for dep in reversed(flat_list) if dep in target_dict['dependencies']]