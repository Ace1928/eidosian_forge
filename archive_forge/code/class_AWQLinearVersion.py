import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
class AWQLinearVersion(str, Enum):
    GEMM = 'gemm'
    GEMV = 'gemv'

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == 'gemm':
            return AWQLinearVersion.GEMM
        elif version == 'gemv':
            return AWQLinearVersion.GEMV
        else:
            raise ValueError(f'Unknown AWQLinearVersion {version}')