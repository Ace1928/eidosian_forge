from __future__ import annotations
import os
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from ..build import CustomTarget
from ..interpreter.type_checking import NoneType, in_set_validator
from ..interpreterbase import typed_pos_args, typed_kwargs, KwargInfo
from ..mesonlib import File, MesonException
class FindProtocol(TypedDict):
    state: Literal['stable', 'staging', 'unstable']
    version: T.Optional[int]