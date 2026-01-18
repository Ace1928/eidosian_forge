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
def _create_lib(cargo: Manifest, build: builder.Builder, crate_type: manifest.CRATE_TYPE) -> T.List[mparser.BaseNode]:
    dependencies: T.List[mparser.BaseNode] = []
    dependency_map: T.Dict[mparser.BaseNode, mparser.BaseNode] = {}
    for name, dep in cargo.dependencies.items():
        package_name = dep.package or name
        dependencies.append(build.identifier(_dependency_varname(package_name)))
        if name != package_name:
            dependency_map[build.string(fixup_meson_varname(package_name))] = build.string(name)
    rust_args: T.List[mparser.BaseNode] = [build.identifier('features_args'), build.identifier(_extra_args_varname())]
    dependencies.append(build.identifier(_extra_deps_varname()))
    posargs: T.List[mparser.BaseNode] = [build.string(fixup_meson_varname(cargo.package.name)), build.string(cargo.lib.path)]
    kwargs: T.Dict[str, mparser.BaseNode] = {'dependencies': build.array(dependencies), 'rust_dependency_map': build.dict(dependency_map), 'rust_args': build.array(rust_args)}
    lib: mparser.BaseNode
    if cargo.lib.proc_macro or crate_type == 'proc-macro':
        lib = build.method('proc_macro', build.identifier('rust'), posargs, kwargs)
    else:
        if crate_type in {'lib', 'rlib', 'staticlib'}:
            target_type = 'static_library'
        elif crate_type in {'dylib', 'cdylib'}:
            target_type = 'shared_library'
        else:
            raise MesonException(f'Unsupported crate type {crate_type}')
        if crate_type in {'staticlib', 'cdylib'}:
            kwargs['rust_abi'] = build.string('c')
        lib = build.function(target_type, posargs, kwargs)
    return [build.assign(build.array([]), 'features_args'), build.foreach(['f', '_'], build.identifier('features'), build.block([build.plusassign(build.array([build.string('--cfg'), build.plus(build.string('feature="'), build.plus(build.identifier('f'), build.string('"')))]), 'features_args')])), build.assign(lib, 'lib'), build.assign(build.function('declare_dependency', kw={'link_with': build.identifier('lib'), 'variables': build.dict({build.string('features'): build.method('join', build.string(','), [build.method('keys', build.identifier('features'))])})}), 'dep'), build.method('override_dependency', build.identifier('meson'), [build.string(_dependency_name(cargo.package.name)), build.identifier('dep')])]