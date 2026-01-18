import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
class QuantizationLiteTestCase(QuantizationTestCase):

    def _create_quantized_model(self, model_class: Type[torch.nn.Module], **kwargs):
        qengine = 'qnnpack'
        with override_quantized_engine(qengine):
            qconfig = torch.ao.quantization.get_default_qconfig(qengine)
            model = model_class(**kwargs)
            model = quantize(model, test_only_eval_fn, [self.calib_data])
        return model

    def _compare_script_and_mobile(self, model: torch.nn.Module, input: torch.Tensor):
        qengine = 'qnnpack'
        with override_quantized_engine(qengine):
            script_module = torch.jit.script(model)
            script_module_result = script_module(input)
            max_retry = 5
            for retry in range(1, max_retry + 1):
                try:
                    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
                    buffer.seek(0)
                    mobile_module = _load_for_lite_interpreter(buffer)
                    mobile_module_result = mobile_module(input)
                    torch.testing.assert_close(script_module_result, mobile_module_result)
                    mobile_module_forward_result = mobile_module.forward(input)
                    torch.testing.assert_close(script_module_result, mobile_module_forward_result)
                    mobile_module_run_method_result = mobile_module.run_method('forward', input)
                    torch.testing.assert_close(script_module_result, mobile_module_run_method_result)
                except AssertionError as e:
                    if retry == max_retry:
                        raise e
                    else:
                        continue
                break