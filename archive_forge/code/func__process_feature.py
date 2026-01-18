from __future__ import annotations
import dataclasses
import glob
import importlib
import itertools
import json
import os
import shutil
import collections
import typing as T
from . import builder
from . import version
from ..mesonlib import MesonException, Popen_safe, OptionKey
from .. import coredata
def _process_feature(cargo: Manifest, feature: str) -> T.Tuple[T.Set[str], T.Dict[str, T.Set[str]], T.Set[str]]:
    features: T.Set[str] = set()
    dep_features: T.Dict[str, T.Set[str]] = collections.defaultdict(set)
    required_deps: T.Set[str] = set()
    to_process: T.Set[str] = {feature}
    while to_process:
        f = to_process.pop()
        if '/' in f:
            dep, dep_f = f.split('/', 1)
            if dep[-1] == '?':
                dep = dep[:-1]
            else:
                required_deps.add(dep)
            dep_features[dep].add(dep_f)
        elif f.startswith('dep:'):
            required_deps.add(f[4:])
        elif f not in features:
            features.add(f)
            to_process.update(cargo.features.get(f, []))
            if f in cargo.dependencies:
                required_deps.add(f)
    return (features, dep_features, required_deps)