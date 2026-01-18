from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def add_deps(self, deps):
    deps = listify(deps)
    for dep in deps:
        if dep in self.added_deps:
            continue
        if isinstance(dep, dependencies.InternalDependency):
            self.process_sourcelist(dep.sources)
            self.extra_files.extend((f for f in dep.extra_files if f not in self.extra_files))
            self.add_include_dirs(dep.include_directories, dep.get_include_type())
            self.objects.extend(dep.objects)
            self.link_targets.extend(dep.libraries)
            self.link_whole_targets.extend(dep.whole_libraries)
            if dep.get_compile_args() or dep.get_link_args():
                extpart = dependencies.InternalDependency('undefined', [], dep.get_compile_args(), dep.get_link_args(), [], [], [], [], [], {}, [], [], [])
                self.external_deps.append(extpart)
            self.add_deps(dep.ext_deps)
        elif isinstance(dep, dependencies.Dependency):
            if dep not in self.external_deps:
                self.external_deps.append(dep)
                self.process_sourcelist(dep.get_sources())
            self.add_deps(dep.ext_deps)
        elif isinstance(dep, BuildTarget):
            raise InvalidArguments(f'Tried to use a build target {dep.name} as a dependency of target {self.name}.\nYou probably should put it in link_with instead.')
        else:
            if hasattr(dep, 'held_object'):
                dep = dep.held_object
            if hasattr(dep, 'project_args_frozen') or hasattr(dep, 'global_args_frozen'):
                raise InvalidArguments('Tried to use subproject object as a dependency.\nYou probably wanted to use a dependency declared in it instead.\nAccess it by calling get_variable() on the subproject object.')
            raise InvalidArguments(f'Argument is of an unacceptable type {type(dep).__name__!r}.\nMust be either an external dependency (returned by find_library() or dependency()) or an internal dependency (returned by declare_dependency()).')
        dep_d_features = dep.d_features
        for feature in ('versions', 'import_dirs'):
            if feature in dep_d_features:
                self.d_features[feature].extend(dep_d_features[feature])
        self.added_deps.add(dep)