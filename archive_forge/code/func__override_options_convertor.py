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
def _override_options_convertor(raw: T.Union[str, T.List[str], T.Dict[str, T.Union[str, int, bool, T.List[str]]]]) -> T.Dict[OptionKey, T.Union[str, int, bool, T.List[str]]]:
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        output: T.Dict[OptionKey, T.Union[str, int, bool, T.List[str]]] = {}
        for each in raw:
            k, v = split_equal_string(each)
            output[OptionKey.from_string(k)] = v
        return output
    return {OptionKey.from_string(k): v for k, v in raw.items()}