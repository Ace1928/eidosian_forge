from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
def _add_model_test(self, model_test: TestCase, kind: str) -> None:
    model_marker: list[ModelProto | NodeProto | None] = [None]

    def run(test_self: Any, device: str, **kwargs) -> None:
        if model_test.url is not None and model_test.url.startswith('onnx/backend/test/data/light/'):
            model_pb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', model_test.url))
            if not os.path.exists(model_pb_path):
                raise FileNotFoundError(f'Unable to find model {model_pb_path!r}.')
            onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
            models_dir = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models', 'light'))
            model_dir: str = os.path.join(models_dir, model_test.model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            use_dummy = True
        else:
            if model_test.model_dir is None:
                model_dir = self.prepare_model_data(model_test)
            else:
                model_dir = model_test.model_dir
            model_pb_path = os.path.join(model_dir, 'model.onnx')
            use_dummy = False
        if not ONNX_ML and 'ai_onnx_ml' in model_dir:
            return
        model = onnx.load(model_pb_path)
        model_marker[0] = model
        if hasattr(self.backend, 'is_compatible') and callable(self.backend.is_compatible) and (not self.backend.is_compatible(model)):
            raise unittest.SkipTest('Not compatible with backend')
        prepared_model = self.backend.prepare(model, device, **kwargs)
        assert prepared_model is not None
        if use_dummy:
            with open(model_pb_path, 'rb') as f:
                onx = onnx.load(f)
            test_data_set = os.path.join(model_dir, 'test_data_set_0')
            if not os.path.exists(test_data_set):
                os.mkdir(test_data_set)
            feeds = {}
            inits = {i.name for i in onx.graph.initializer}
            n_input = 0
            inputs = []
            for i in range(len(onx.graph.input)):
                if onx.graph.input[i].name in inits:
                    continue
                name = os.path.join(test_data_set, f'input_{n_input}.pb')
                inputs.append(name)
                n_input += 1
                x = onx.graph.input[i]
                value = self.generate_dummy_data(x, seed=0, name=model_test.model_name, random=False)
                feeds[x.name] = value
                with open(name, 'wb') as f:
                    f.write(onnx.numpy_helper.from_array(value).SerializeToString())
            prefix = os.path.splitext(model_pb_path)[0]
            expected_outputs = []
            for i in range(len(onx.graph.output)):
                name = f'{prefix}_output_{i}.pb'
                if os.path.exists(name):
                    expected_outputs.append(name)
                    continue
                expected_outputs = None
                break
            if expected_outputs is None:
                ref = onnx.reference.ReferenceEvaluator(onx)
                outputs = ref.run(None, feeds)
                for i, o in enumerate(outputs):
                    name = os.path.join(test_data_set, f'output_{i}.pb')
                    with open(name, 'wb') as f:
                        f.write(onnx.numpy_helper.from_array(o).SerializeToString())
            else:
                for i, o in enumerate(expected_outputs):
                    name = os.path.join(test_data_set, f'output_{i}.pb')
                    shutil.copy(o, name)
        else:
            for test_data_npz in glob.glob(os.path.join(model_dir, 'test_data_*.npz')):
                test_data = np.load(test_data_npz, encoding='bytes')
                inputs = list(test_data['inputs'])
                outputs = list(prepared_model.run(inputs))
                ref_outputs = tuple((np.array(x) if not isinstance(x, (list, dict)) else x for f in test_data['outputs']))
                self.assert_similar_outputs(ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol, model_dir=model_dir)
        for test_data_dir in glob.glob(os.path.join(model_dir, 'test_data_set*')):
            inputs = []
            inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
            for i in range(inputs_num):
                input_file = os.path.join(test_data_dir, f'input_{i}.pb')
                self._load_proto(input_file, inputs, model.graph.input[i].type)
            ref_outputs = []
            ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
            for i in range(ref_outputs_num):
                output_file = os.path.join(test_data_dir, f'output_{i}.pb')
                self._load_proto(output_file, ref_outputs, model.graph.output[i].type)
            outputs = list(prepared_model.run(inputs))
            self.assert_similar_outputs(ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol, model_dir=model_dir)
    if model_test.name in self._test_kwargs:
        self._add_test(kind + 'Model', model_test.name, run, model_marker, **self._test_kwargs[model_test.name])
    else:
        self._add_test(kind + 'Model', model_test.name, run, model_marker)