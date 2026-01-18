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
def _options_validator(value: T.Union[EnvironmentVariables, T.List['TYPE_var'], T.Dict[str, 'TYPE_var'], str, None]) -> T.Optional[str]:
    return _env_validator(value, only_dict_str=False)