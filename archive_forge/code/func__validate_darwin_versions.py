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
def _validate_darwin_versions(darwin_versions: T.List[T.Union[str, int]]) -> T.Optional[str]:
    if len(darwin_versions) > 2:
        return f'Must contain between 0 and 2 elements, not {len(darwin_versions)}'
    if len(darwin_versions) == 1:
        darwin_versions = 2 * darwin_versions
    for v in darwin_versions:
        if isinstance(v, int):
            v = str(v)
        if not re.fullmatch('[0-9]+(\\.[0-9]+){0,2}', v):
            return 'must be X.Y.Z where X, Y, Z are numbers, and Y and Z are optional'
        try:
            parts = v.split('.')
        except ValueError:
            return f'badly formed value: "{v}, not in X.Y.Z form'
        if len(parts) in {1, 2, 3} and int(parts[0]) > 65535:
            return 'must be X.Y.Z where X is [0, 65535] and Y, Z are optional'
        if len(parts) in {2, 3} and int(parts[1]) > 255:
            return 'must be X.Y.Z where Y is [0, 255] and Y, Z are optional'
        if len(parts) == 3 and int(parts[2]) > 255:
            return 'must be X.Y.Z where Z is [0, 255] and Y, Z are optional'
    return None