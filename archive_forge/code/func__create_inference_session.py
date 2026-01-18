from __future__ import annotations
import platform
import unittest
from typing import Any
import numpy
from packaging.version import Version
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx.backend.base import Device, DeviceType
def _create_inference_session(model: onnx.ModelProto, device: str):
    if device == 'CPU':
        providers = ('CPUExecutionProvider',)
    elif device == 'CUDA':
        providers = ('CUDAExecutionProvider',)
    else:
        raise ValueError(f'Unexpected device {device!r}.')
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    except Exception as e:
        raise RuntimeError(f'Unable to create inference session. Model is:\n\n{onnx.printer.to_text(model)}') from e
    return session