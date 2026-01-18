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
class AutoOptimizationConfig:
    """
    Factory to create common `OptimizationConfig`.
    """
    _LEVELS = {'O1': {'optimization_level': 1, 'enable_transformers_specific_optimizations': False}, 'O2': {'optimization_level': 2, 'enable_transformers_specific_optimizations': True}, 'O3': {'optimization_level': 2, 'enable_transformers_specific_optimizations': True, 'enable_gelu_approximation': True}, 'O4': {'optimization_level': 2, 'enable_transformers_specific_optimizations': True, 'enable_gelu_approximation': True, 'fp16': True}}

    @classmethod
    def with_optimization_level(cls, optimization_level: str, for_gpu: bool=False, **kwargs) -> OptimizationConfig:
        """
        Creates an [`~OptimizationConfig`] with pre-defined arguments according to an optimization level.

        Args:
            optimization_level (`str`):
                The optimization level, the following values are allowed:
                - O1: Basic general optimizations
                - O2: Basic and extended general optimizations, transformers-specific fusions.
                - O3: Same as O2 with Fast Gelu approximation.
                - O4: Same as O3 with mixed precision.
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the requested optimization level.
        """
        if optimization_level not in cls._LEVELS:
            raise ValueError(f'optimization_level must be in {', '.join(cls._LEVELS.keys())}, got {optimization_level}')
        if optimization_level == 'O4':
            if for_gpu is False:
                logger.warning('Overridding for_gpu=False to for_gpu=True as half precision is available only on GPU.')
            for_gpu = True
        return OptimizationConfig(optimize_for_gpu=for_gpu, **cls._LEVELS[optimization_level], **kwargs)

    @classmethod
    def O1(cls, for_gpu: bool=False, **kwargs) -> OptimizationConfig:
        """
        Creates an O1 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O1 optimization level.
        """
        return cls.with_optimization_level('O1', for_gpu=for_gpu, **kwargs)

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

    @classmethod
    def O3(cls, for_gpu: bool=False, **kwargs) -> OptimizationConfig:
        """
        Creates an O3 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O3 optimization level.
        """
        return cls.with_optimization_level('O3', for_gpu=for_gpu, **kwargs)

    @classmethod
    def O4(cls, for_gpu: bool=True, **kwargs) -> OptimizationConfig:
        """
        Creates an O4 [`~OptimizationConfig`].

        Args:
            for_gpu (`bool`, defaults to `False`):
                Whether the model to optimize will run on GPU, some optimizations depends on the hardware the model
                will run on. Only needed for optimization_level > 1.
            kwargs (`Dict[str, Any]`):
                Arguments to provide to the [`~OptimizationConfig`] constructor.

        Returns:
            `OptimizationConfig`: The `OptimizationConfig` corresponding to the O4 optimization level.
        """
        return cls.with_optimization_level('O4', for_gpu=for_gpu, **kwargs)