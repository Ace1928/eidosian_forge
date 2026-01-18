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
def _validate_win_subsystem(value: T.Optional[str]) -> T.Optional[str]:
    if value is not None:
        if re.fullmatch('(boot_application|console|efi_application|efi_boot_service_driver|efi_rom|efi_runtime_driver|native|posix|windows)(,\\d+(\\.\\d+)?)?', value) is None:
            return f'Invalid value for win_subsystem: {value}.'
    return None