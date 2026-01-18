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
def _output_validator(outputs: T.List[str]) -> T.Optional[str]:
    output_set = set(outputs)
    if len(output_set) != len(outputs):
        seen = set()
        for el in outputs:
            if el in seen:
                return f'contains {el!r} multiple times, but no duplicates are allowed.'
            seen.add(el)
    for i in outputs:
        if i == '':
            return 'Output must not be empty.'
        elif i.strip() == '':
            return 'Output must not consist only of whitespace.'
        elif has_path_sep(i):
            return f'Output {i!r} must not contain a path segment.'
        elif '@INPUT' in i:
            return f'output {i!r} contains "@INPUT", which is invalid. Did you mean "@PLAINNAME@" or "@BASENAME@?'
    return None