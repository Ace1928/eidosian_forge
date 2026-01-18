from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
def _create_dummy_input_generator_classes(self) -> List['DummyInputGenerator']:
    self._validate_mandatory_axes()
    return [cls_(self.task, self._normalized_config, **self._axes) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]