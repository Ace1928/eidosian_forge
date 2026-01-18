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
def _name_validator(arg: T.Optional[T.Union[str, T.List]]) -> T.Optional[str]:
    if isinstance(arg, list) and arg:
        return 'must be empty when passed as an array to signify the default value.'
    return None