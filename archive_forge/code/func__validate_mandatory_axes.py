from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
def _validate_mandatory_axes(self):
    for name, axis_dim in self._axes.items():
        if axis_dim is None:
            raise MissingMandatoryAxisDimension(f'The value for the {name} axis is missing, it is needed to perform the export to TensorFlow Lite.')