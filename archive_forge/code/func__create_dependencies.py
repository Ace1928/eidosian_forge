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
def _create_dependencies(cargo: Manifest, build: builder.Builder) -> T.List[mparser.BaseNode]:
    ast: T.List[mparser.BaseNode] = []
    for name, dep in cargo.dependencies.items():
        package_name = dep.package or name
        extra_options: T.Dict[mparser.BaseNode, mparser.BaseNode] = {build.string(_option_name('default')): build.bool(dep.default_features)}
        for f in dep.features:
            extra_options[build.string(_option_name(f))] = build.bool(True)
        ast.append(build.plusassign(build.dict(extra_options), _options_varname(name)))
        kw = {'version': build.array([build.string(s) for s in dep.version]), 'default_options': build.identifier(_options_varname(name))}
        if dep.optional:
            kw['required'] = build.method('get', build.identifier('required_deps'), [build.string(name), build.bool(False)])
        ast.extend([build.assign(build.function('dependency', [build.string(_dependency_name(package_name))], kw), _dependency_varname(package_name)), build.if_(build.method('found', build.identifier(_dependency_varname(package_name))), build.block([build.assign(build.method('split', build.method('get_variable', build.identifier(_dependency_varname(package_name)), [build.string('features')], {'default_value': build.string('')}), [build.string(',')]), 'actual_features'), build.assign(build.array([]), 'needed_features'), build.foreach(['f', 'enabled'], build.identifier(_options_varname(name)), build.block([build.if_(build.identifier('enabled'), build.block([build.plusassign(build.method('substring', build.identifier('f'), [build.number(len(_OPTION_NAME_PREFIX))]), 'needed_features')]))])), build.foreach(['f'], build.identifier('needed_features'), build.block([build.if_(build.not_in(build.identifier('f'), build.identifier('actual_features')), build.block([build.function('error', [build.string('Dependency'), build.string(_dependency_name(package_name)), build.string('previously configured with features'), build.identifier('actual_features'), build.string('but need'), build.identifier('needed_features')])]))]))]))])
    return ast