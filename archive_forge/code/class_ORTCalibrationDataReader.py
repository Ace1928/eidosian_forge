import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union
import onnx
from datasets import Dataset, load_dataset
from packaging.version import Version, parse
from transformers import AutoConfig
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from ..quantization_base import OptimumQuantizer
from ..utils.save_utils import maybe_save_preprocessors
from . import ORTQuantizableOperator
from .configuration import CalibrationConfig, ORTConfig, QuantizationConfig
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .preprocessors import QuantizationPreprocessor
class ORTCalibrationDataReader(CalibrationDataReader):
    __slots__ = ['batch_size', 'dataset', '_dataset_iter']

    def __init__(self, dataset: Dataset, batch_size: int=1):
        if dataset is None:
            raise ValueError('Provided dataset is None.')
        if batch_size <= 0:
            raise ValueError(f'Provided batch_size should be >= 1 (got: {batch_size}).')
        self.dataset = dataset
        self.batch_size = batch_size
        self._dataset_iter = iter(self.dataset)

    def get_next(self):
        featurized_samples = None
        try:
            if self.batch_size == 1:
                featurized_samples = {key: [value] for key, value in next(self._dataset_iter).items()}
            else:
                featurized_samples = defaultdict(list)
                for _ in range(self.batch_size):
                    sample = next(self._dataset_iter)
                    for name, value in sample.items():
                        featurized_samples[name] += [value]
        except StopIteration:
            pass
        if featurized_samples is not None and len(featurized_samples) > 0:
            return featurized_samples
        return None