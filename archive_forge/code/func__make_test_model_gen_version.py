import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def _make_test_model_gen_version(graph: GraphProto, **kwargs: Any) -> ModelProto:
    latest_onnx_version, latest_ml_version, latest_training_version = onnx.helper.VERSION_TABLE[-1][2:5]
    if 'opset_imports' in kwargs:
        for opset in kwargs['opset_imports']:
            if opset.domain in {'', 'ai.onnx'} and opset.version == latest_onnx_version + 1 or (opset.domain == 'ai.onnx.ml' and opset.version == latest_ml_version + 1) or (opset.domain in {'ai.onnx.training version', 'ai.onnx.preview.training'} and opset.version == latest_training_version + 1):
                return onnx.helper.make_model(graph, **kwargs)
    return onnx.helper.make_model_gen_version(graph, **kwargs)