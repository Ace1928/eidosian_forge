from __future__ import annotations
import itertools, os, re
import typing as T
from .. import compilers
from ..build import (CustomTarget, BuildTarget,
from ..coredata import UserFeatureOption
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase.decorators import KwargInfo, ContainerTypeInfo
from ..mesonlib import (File, FileMode, MachineChoice, listify, has_path_sep,
from ..programs import ExternalProgram
def _env_validator(value: T.Union[EnvironmentVariables, T.List['TYPE_var'], T.Dict[str, 'TYPE_var'], str, None], only_dict_str: bool=True) -> T.Optional[str]:

    def _splitter(v: str) -> T.Optional[str]:
        split = v.split('=', 1)
        if len(split) == 1:
            return f'"{v}" is not two string values separated by an "="'
        return None
    if isinstance(value, str):
        v = _splitter(value)
        if v is not None:
            return v
    elif isinstance(value, list):
        for i in listify(value):
            if not isinstance(i, str):
                return f'All array elements must be a string, not {i!r}'
            v = _splitter(i)
            if v is not None:
                return v
    elif isinstance(value, dict):
        for k, dv in value.items():
            if only_dict_str:
                if any((i for i in listify(dv) if not isinstance(i, str))):
                    return f'Dictionary element {k} must be a string or list of strings not {dv!r}'
            elif isinstance(dv, list):
                if any((not isinstance(i, str) for i in dv)):
                    return f'Dictionary element {k} must be a string, bool, integer or list of strings, not {dv!r}'
            elif not isinstance(dv, (str, bool, int)):
                return f'Dictionary element {k} must be a string, bool, integer or list of strings, not {dv!r}'
    return None