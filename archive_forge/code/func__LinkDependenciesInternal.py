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
def _LinkDependenciesInternal(self, targets, include_shared_libraries, dependencies=None, initial=True):
    """Returns an OrderedSet of dependency targets that are linked
    into this target.

    This function has a split personality, depending on the setting of
    |initial|.  Outside callers should always leave |initial| at its default
    setting.

    When adding a target to the list of dependencies, this function will
    recurse into itself with |initial| set to False, to collect dependencies
    that are linked into the linkable target for which the list is being built.

    If |include_shared_libraries| is False, the resulting dependencies will not
    include shared_library targets that are linked into this target.
    """
    if dependencies is None:
        dependencies = OrderedSet()
    if self.ref is None:
        return dependencies
    if 'target_name' not in targets[self.ref]:
        raise GypError("Missing 'target_name' field in target.")
    if 'type' not in targets[self.ref]:
        raise GypError("Missing 'type' field in target %s" % targets[self.ref]['target_name'])
    target_type = targets[self.ref]['type']
    is_linkable = target_type in linkable_types
    if initial and (not is_linkable):
        return dependencies
    if target_type == 'none' and (not targets[self.ref].get('dependencies_traverse', True)):
        dependencies.add(self.ref)
        return dependencies
    if not initial and target_type in ('executable', 'loadable_module', 'mac_kernel_extension', 'windows_driver'):
        return dependencies
    if not initial and target_type == 'shared_library' and (not include_shared_libraries):
        return dependencies
    if self.ref not in dependencies:
        dependencies.add(self.ref)
        if initial or not is_linkable:
            for dependency in self.dependencies:
                dependency._LinkDependenciesInternal(targets, include_shared_libraries, dependencies, False)
    return dependencies