import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
class TestGradients(TestCase):
    exact_dtype = True

    def _get_safe_inplace(self, inplace_variant):

        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)
        return _fn

    def _check_helper(self, device, dtype, op, variant, check, *, check_forward_ad=False, check_backward_ad=True, check_batched_grad=None, check_batched_forward_grad=False):
        assert check in ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad')
        if variant is None:
            self.skipTest('Skipped! Variant not implemented.')
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f'Skipped! {op.name} does not support dtype {str(dtype)}')

        def is_inplace(variant):
            if hasattr(variant, '__wrapped__'):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()
        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs, small_inputs_only=TEST_WITH_SLOW_GRADCHECK)
        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue
            all_args = None
            if is_iterable_of_tensors(sample.input):
                all_args = chain(sample.input, sample.args, sample.kwargs.values())
            else:
                all_args = tuple(chain((sample.input,), sample.args, sample.kwargs.values()))
            gradcheck_args = tuple((x for x in all_args if isinstance(x, torch.Tensor) and x.requires_grad))
            for t in gradcheck_args:
                self.assertIsNone(t.grad, 'A sampled input has a gradient before running autograd. This usually means that (at least) one input tensor is reused across different SampleInputs. Please create a new tensor for each SampleInput.')

            def _input_recomposition_helper(inputs, inp, input_idx):
                if is_iterable_of_tensors(inp):
                    tensor_list = []
                    for x in inp:
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            tensor_list.append(inputs[input_idx])
                            input_idx = input_idx + 1
                        else:
                            tensor_list.append(x)
                    return (tensor_list, input_idx)
                elif isinstance(inp, torch.Tensor) and inp.requires_grad:
                    return (inputs[input_idx], input_idx + 1)
                else:
                    return (inp, input_idx)

            def fn(*inputs):
                positional_args = []
                input_idx = 0
                inp, input_idx = _input_recomposition_helper(inputs, sample.input, input_idx)
                positional_args.append(inp)
                for x in sample.args:
                    inp, input_idx = _input_recomposition_helper(inputs, x, input_idx)
                    positional_args.append(inp)
                kwargs = {}
                for k, v in sample.kwargs.items():
                    inp, input_idx = _input_recomposition_helper(inputs, v, input_idx)
                    kwargs[k] = inp
                output = op.gradcheck_wrapper(variant, *positional_args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output
            if check == 'gradcheck':
                if check_batched_grad is None:
                    check_batched_grad = op.check_batched_grad
                self.assertTrue(gradcheck(fn, gradcheck_args, check_batched_grad=check_batched_grad, check_grad_dtypes=True, nondet_tol=op.gradcheck_nondet_tol, fast_mode=op.gradcheck_fast_mode, check_forward_ad=check_forward_ad, check_backward_ad=check_backward_ad, check_undefined_grad=True, check_batched_forward_grad=check_batched_forward_grad))
            elif check in ('bwgrad_bwgrad', 'fwgrad_bwgrad'):
                self.assertFalse(check_forward_ad, msg='Cannot run forward AD check for gradgradcheck')
                for gen_non_contig_grad_outputs in (False, True):
                    kwargs = {'gen_non_contig_grad_outputs': gen_non_contig_grad_outputs, 'check_batched_grad': op.check_batched_gradgrad, 'check_grad_dtypes': True, 'nondet_tol': op.gradcheck_nondet_tol, 'fast_mode': op.gradcheck_fast_mode}
                    if check == 'fwgrad_bwgrad':
                        kwargs['check_fwd_over_rev'] = True
                        kwargs['check_rev_over_rev'] = False
                        kwargs['check_batched_grad'] = False
                        kwargs['check_undefined_grad'] = False
                    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
            else:
                self.assertTrue(False, msg='Unknown check requested!')

    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True, check_batched_grad=None, check_batched_forward_grad=False):
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad, check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad, check_batched_forward_grad=check_batched_forward_grad)

    def _skip_helper(self, op, device, dtype):
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Op doesn't support autograd for this dtype.")
        if not op.supports_autograd and (not op.supports_forward_ad):
            self.skipTest('Skipped! autograd not supported.')