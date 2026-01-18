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
def _do_dependency(self, kwargs: TYPE_nkwargs, func_args: TYPE_nvar, func_kwargs: TYPE_nkwargs) -> T.Optional[Dependency]:
    name = func_args[0]
    self._handle_featurenew_dependencies(name)
    dep = dependencies.find_external_dependency(name, self.environment, kwargs)
    if dep.found():
        for_machine = self.interpreter.machine_from_native_kwarg(kwargs)
        identifier = dependencies.get_dep_identifier(name, kwargs)
        self.coredata.deps[for_machine].put(identifier, dep)
        return dep
    return None