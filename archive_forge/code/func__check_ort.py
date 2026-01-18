import itertools
import unittest
from functools import wraps
from os import getenv
import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized
import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
@staticmethod
def _check_ort(onx, feeds, atol=0, rtol=0, equal=False, rev=False):
    if not has_onnxruntime():
        return
    from onnxruntime import InferenceSession
    onnx_domain_opset = ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION
    ml_domain_opset = ORT_MAX_ML_OPSET_SUPPORTED_VERSION
    for opset in onx.opset_import:
        if opset.domain in ('', 'ai.onnx'):
            onnx_domain_opset = opset.version
            break
    for opset in onx.opset_import:
        if opset.domain == 'ai.onnx.ml':
            ml_domain_opset = opset.version
            break
    if onx.ir_version > ORT_MAX_IR_SUPPORTED_VERSION or onnx_domain_opset > ORT_MAX_ONNX_OPSET_SUPPORTED_VERSION or ml_domain_opset > ORT_MAX_ML_OPSET_SUPPORTED_VERSION:
        return
    ort = InferenceSession(onx.SerializeToString(), providers=['CPUExecutionProvider'])
    sess = ReferenceEvaluator(onx)
    expected = ort.run(None, feeds)
    got = sess.run(None, feeds)
    if len(expected) != len(got):
        raise AssertionError(f'onnxruntime returns a different number of output {len(expected)} != {len(sess)} (ReferenceEvaluator).')
    look = zip(reversed(expected), reversed(got)) if rev else zip(expected, got)
    for i, (e, g) in enumerate(look):
        if e.shape != g.shape:
            raise AssertionError(f'Unexpected shape {g.shape} for output {i} (expecting {e.shape})\n{e!r}\n---\n{g!r}.')
        if equal:
            if e.tolist() != g.tolist():
                raise AssertionError(f'Discrepancies for output {i}\nexpected=\n{e}\n!=\nresults=\n{g}')
        else:
            assert_allclose(actual=g, desired=e, atol=atol, rtol=rtol, err_msg=f'Discrepancies for output {i} expected[0]={e.ravel()[0]}.')