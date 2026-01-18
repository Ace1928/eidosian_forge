import os
import platform
import sys
import unittest
from typing import Any
import numpy
import version_utils
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator
class ReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):

    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(self._session.input_names, self._session.input_types):
                    shape = tuple((d.dim_value for d in tshape.tensor_type.shape.dim))
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f'Unexpected input type {type(inputs)!r}.')
        outs = self._session.run(None, feeds)
        return outs