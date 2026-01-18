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
class DependencyGraphNode:
    """

  Attributes:
    ref: A reference to an object that this DependencyGraphNode represents.
    dependencies: List of DependencyGraphNodes on which this one depends.
    dependents: List of DependencyGraphNodes that depend on this one.
  """

    class CircularException(GypError):
        pass

    def __init__(self, ref):
        self.ref = ref
        self.dependencies = []
        self.dependents = []

    def __repr__(self):
        return '<DependencyGraphNode: %r>' % self.ref

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

    def DirectDependencies(self, dependencies=None):
        """Returns a list of just direct dependencies."""
        if dependencies is None:
            dependencies = []
        for dependency in self.dependencies:
            if dependency.ref and dependency.ref not in dependencies:
                dependencies.append(dependency.ref)
        return dependencies

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

    def DirectAndImportedDependencies(self, targets, dependencies=None):
        """Returns a list of a target's direct dependencies and all indirect
    dependencies that a dependency has advertised settings should be exported
    through the dependency for.
    """
        dependencies = self.DirectDependencies(dependencies)
        return self._AddImportedDependencies(targets, dependencies)

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

    def DependenciesForLinkSettings(self, targets):
        """
    Returns a list of dependency targets whose link_settings should be merged
    into this target.
    """
        include_shared_libraries = targets[self.ref].get('allow_sharedlib_linksettings_propagation', True)
        return self._LinkDependenciesInternal(targets, include_shared_libraries)

    def DependenciesToLinkAgainst(self, targets):
        """
    Returns a list of dependency targets that are linked into this target.
    """
        return self._LinkDependenciesInternal(targets, True)