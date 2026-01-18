import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from packaging.version import Version, parse
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from onnxruntime.transformers.fusion_options import FusionOptions
from ..configuration_utils import BaseConfig
from ..utils import logging
@classmethod
def O2(cls, for_gpu: bool=False, **kwargs) -> OptimizationConfig:
    """
        Creates an O2 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O2 optimization level.
        """
    return cls.with_optimization_level('O2', for_gpu=for_gpu, **kwargs)