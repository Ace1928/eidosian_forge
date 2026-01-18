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
def _do_existing_subproject(self, kwargs: TYPE_nkwargs, func_args: TYPE_nvar, func_kwargs: TYPE_nkwargs) -> T.Optional[Dependency]:
    subp_name = func_args[0]
    varname = self.subproject_varname
    if subp_name and self._get_subproject(subp_name):
        return self._get_subproject_dep(subp_name, varname, kwargs)
    return None