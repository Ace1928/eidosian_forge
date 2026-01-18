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
def _validate_shlib_version(val: T.Optional[str]) -> T.Optional[str]:
    if val is not None and (not re.fullmatch('[0-9]+(\\.[0-9]+){0,2}', val)):
        return f'Invalid Shared library version "{val}". Must be of the form X.Y.Z where all three are numbers. Y and Z are optional.'
    return None