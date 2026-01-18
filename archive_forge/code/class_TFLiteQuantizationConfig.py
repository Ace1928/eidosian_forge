from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
@dataclass
class TFLiteQuantizationConfig:
    """
    Contains all the information needed to perform quantization with TFLite.

    Attributes:

        approach (`Optional[Union[str, QuantizationApproach]]`, defaults to `None`):
            The quantization approach to perform. No quantization is applied if left unspecified.
        fallback_to_float (`bool`, defaults to `False`):
            Allows to fallback to float kernels in quantization.
        inputs_dtype (`Optional[str]`, defaults to `None`):
            The data type of the inputs. If specified it must be either "int8" or "uint8". It allows to always take
            integers as inputs, it is useful for integer-only hardware.
        outputs_dtype (`Optional[str]`, defaults to `None`):
            The data type of the outputs. If specified it must be either "int8" or "uint8". It allows to always output
            integers, it is useful for integer-only hardware.
        calibration_dataset_name_or_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The dataset to use for calibrating the quantization parameters for static quantization. If left unspecified,
            a default dataset for the considered task will be used.
        calibration_dataset_config_name (`Optional[str]`, defaults to `None`):
            The configuration name of the dataset if needed.
        num_calibration_samples (`int`, defaults to `200`):
            The number of examples from the calibration dataset to use to compute the quantization parameters.
        calibration_split (`Optional[str]`, defaults to `None`):
            The split of the dataset to use. If none is specified and the dataset contains multiple splits, the
            smallest split will be used.
        primary_key (`Optional[str]`, defaults `None`):
            The name of the column in the dataset containing the main data to preprocess. Only for
            text-classification and token-classification.
        secondary_key (`Optional[str]`, defaults `None`):
            The name of the second column in the dataset containing the main data to preprocess, not always needed.
            Only for text-classification and token-classification.
        question_key (`Optional[str]`, defaults `None`):
            The name of the column containing the question in the dataset. Only for question-answering.
        context_key (`Optional[str]`, defaults `None`):
            The name of the column containing the context in the dataset. Only for question-answering.
        image_key (`Optional[str]`, defaults `None`):
            The name of the column containing the image in the dataset. Only for image-classification.
    """
    approach: Optional[Union[str, QuantizationApproach]] = None
    fallback_to_float: bool = False
    inputs_dtype: Optional[str] = None
    outputs_dtype: Optional[str] = None
    calibration_dataset_name_or_path: Optional[Union[str, Path]] = None
    calibration_dataset_config_name: Optional[str] = None
    num_calibration_samples: int = 200
    calibration_split: Optional[str] = None
    primary_key: Optional[str] = None
    secondary_key: Optional[str] = None
    question_key: Optional[str] = None
    context_key: Optional[str] = None
    image_key: Optional[str] = None

    def __post_init__(self):
        if self.approach is not None:
            if isinstance(self.approach, str) and (not isinstance(self.approach, QuantizationApproach)):
                self.approach = QuantizationApproach(self.approach)