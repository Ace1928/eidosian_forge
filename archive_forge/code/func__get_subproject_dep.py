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
def _get_subproject_dep(self, subp_name: str, varname: str, kwargs: TYPE_nkwargs) -> T.Optional[Dependency]:
    subproject = self._get_subproject(subp_name)
    if not subproject:
        mlog.log('Dependency', mlog.bold(self._display_name), 'from subproject', mlog.bold(subp_name), 'found:', mlog.red('NO'), mlog.blue('(subproject failed to configure)'))
        return None
    cached_dep = None
    for name in self.names:
        cached_dep = self._get_cached_dep(name, kwargs)
        if cached_dep:
            break
    if cached_dep:
        self._verify_fallback_consistency(cached_dep)
        return cached_dep
    if not varname:
        for name in self.names:
            varname = self.wrap_resolver.get_varname(subp_name, name)
            if varname:
                break
    if not varname:
        mlog.warning(f'Subproject {subp_name!r} did not override {self._display_name!r} dependency and no variable name specified')
        mlog.log('Dependency', mlog.bold(self._display_name), 'from subproject', mlog.bold(subproject.subdir), 'found:', mlog.red('NO'))
        return self._notfound_dependency()
    var_dep = self._get_subproject_variable(subproject, varname) or self._notfound_dependency()
    if not var_dep.found():
        mlog.log('Dependency', mlog.bold(self._display_name), 'from subproject', mlog.bold(subproject.subdir), 'found:', mlog.red('NO'))
        return var_dep
    wanted = stringlistify(kwargs.get('version', []))
    found = var_dep.get_version()
    if not self._check_version(wanted, found):
        mlog.log('Dependency', mlog.bold(self._display_name), 'from subproject', mlog.bold(subproject.subdir), 'found:', mlog.red('NO'), 'found', mlog.normal_cyan(found), 'but need:', mlog.bold(', '.join([f"'{e}'" for e in wanted])))
        return self._notfound_dependency()
    mlog.log('Dependency', mlog.bold(self._display_name), 'from subproject', mlog.bold(subproject.subdir), 'found:', mlog.green('YES'), mlog.normal_cyan(found) if found else None)
    return var_dep