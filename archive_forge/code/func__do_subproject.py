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
def _do_subproject(self, kwargs: TYPE_nkwargs, func_args: TYPE_nvar, func_kwargs: TYPE_nkwargs) -> T.Optional[Dependency]:
    if self.forcefallback:
        mlog.log('Looking for a fallback subproject for the dependency', mlog.bold(self._display_name), 'because:\nUse of fallback dependencies is forced.')
    elif self.nofallback:
        mlog.log('Not looking for a fallback subproject for the dependency', mlog.bold(self._display_name), 'because:\nUse of fallback dependencies is disabled.')
        return None
    else:
        mlog.log('Looking for a fallback subproject for the dependency', mlog.bold(self._display_name))
    static = kwargs.get('static')
    default_options = func_kwargs.get('default_options', {})
    if static is not None and 'default_library' not in default_options:
        default_library = 'static' if static else 'shared'
        mlog.log(f'Building fallback subproject with default_library={default_library}')
        default_options[OptionKey('default_library')] = default_library
        func_kwargs['default_options'] = default_options
    subp_name = self.subproject_name
    varname = self.subproject_varname
    func_kwargs.setdefault('version', [])
    if 'default_options' in kwargs and isinstance(kwargs['default_options'], str):
        func_kwargs['default_options'] = listify(kwargs['default_options'])
    self.interpreter.do_subproject(subp_name, func_kwargs)
    return self._get_subproject_dep(subp_name, varname, kwargs)