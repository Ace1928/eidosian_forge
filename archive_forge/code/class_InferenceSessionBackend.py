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
class InferenceSessionBackend(onnx.backend.base.Backend):

    @classmethod
    def supports_device(cls, device: str) -> bool:
        providers = set(ort.get_available_providers())
        d = Device(device)
        if d.type == DeviceType.CPU and 'CPUExecutionProvider' in providers:
            return True
        if d.type == DeviceType.CUDA and 'CUDAExecutionProvider' in providers:
            return True
        return False

    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str='CPU', **kwargs: Any) -> InferenceSessionBackendRep:
        del kwargs
        if not isinstance(model, (str, bytes, onnx.ModelProto)):
            raise TypeError(f'Unexpected type {type(model)} for model.')
        session = _create_inference_session(model, device)
        return InferenceSessionBackendRep(session)

    @classmethod
    def run_model(cls, model: onnx.ModelProto, inputs, device=None, **kwargs):
        return super().run_model(model, inputs, device=device, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError('Unable to run the model node by node.')