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
def _AddImportedDependencies(self, targets, dependencies=None):
    """Given a list of direct dependencies, adds indirect dependencies that
    other dependencies have declared to export their settings.

    This method does not operate on self.  Rather, it operates on the list
    of dependencies in the |dependencies| argument.  For each dependency in
    that list, if any declares that it exports the settings of one of its
    own dependencies, those dependencies whose settings are "passed through"
    are added to the list.  As new items are added to the list, they too will
    be processed, so it is possible to import settings through multiple levels
    of dependencies.

    This method is not terribly useful on its own, it depends on being
    "primed" with a list of direct dependencies such as one provided by
    DirectDependencies.  DirectAndImportedDependencies is intended to be the
    public entry point.
    """
    if dependencies is None:
        dependencies = []
    index = 0
    while index < len(dependencies):
        dependency = dependencies[index]
        dependency_dict = targets[dependency]
        add_index = 1
        for imported_dependency in dependency_dict.get('export_dependent_settings', []):
            if imported_dependency not in dependencies:
                dependencies.insert(index + add_index, imported_dependency)
                add_index = add_index + 1
        index = index + 1
    return dependencies