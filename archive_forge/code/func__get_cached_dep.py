from __future__ import annotations
from .interpreterobjects import extract_required_kwarg
from .. import mlog
from .. import dependencies
from .. import build
from ..wrap import WrapMode
from ..mesonlib import OptionKey, extract_as_list, stringlistify, version_compare_many, listify
from ..dependencies import Dependency, DependencyException, NotFoundDependency
from ..interpreterbase import (MesonInterpreterObject, FeatureNew,
import typing as T
def _get_cached_dep(self, name: str, kwargs: TYPE_nkwargs) -> T.Optional[Dependency]:
    for_machine = self.interpreter.machine_from_native_kwarg(kwargs)
    identifier = dependencies.get_dep_identifier(name, kwargs)
    wanted_vers = stringlistify(kwargs.get('version', []))
    override = self.build.dependency_overrides[for_machine].get(identifier)
    if override:
        info = [mlog.blue('(overridden)' if override.explicit else '(cached)')]
        cached_dep = override.dep
        if not cached_dep.found():
            mlog.log('Dependency', mlog.bold(self._display_name), 'found:', mlog.red('NO'), *info)
            return cached_dep
    elif self.forcefallback and self.subproject_name:
        cached_dep = None
    else:
        info = [mlog.blue('(cached)')]
        cached_dep = self.coredata.deps[for_machine].get(identifier)
    if cached_dep:
        found_vers = cached_dep.get_version()
        if not self._check_version(wanted_vers, found_vers):
            if not override:
                return None
            mlog.log('Dependency', mlog.bold(name), 'found:', mlog.red('NO'), 'found', mlog.normal_cyan(found_vers), 'but need:', mlog.bold(', '.join([f"'{e}'" for e in wanted_vers])), *info)
            return self._notfound_dependency()
        if found_vers:
            info = [mlog.normal_cyan(found_vers), *info]
        mlog.log('Dependency', mlog.bold(self._display_name), 'found:', mlog.green('YES'), *info)
        return cached_dep
    return None