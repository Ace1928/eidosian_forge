from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
class QuantizationApproach(str, Enum):
    INT8_DYNAMIC = 'int8-dynamic'
    INT8 = 'int8'
    INT8x16 = 'int8x16'
    FP16 = 'fp16'