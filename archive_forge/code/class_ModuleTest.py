from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
class ModuleTest(TestBase):

    @abstractmethod
    def _do_test(self, test_case: Any, module: nn.Module, input: Any) -> Any:
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 0.0002)
        self.check_forward_only = kwargs.get('check_forward_only', False)
        self.default_dtype = kwargs.get('default_dtype', None)
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()

    def __call__(self, test_case):
        with set_default_dtype(self.default_dtype):
            module = self.constructor(*self.constructor_args)
            input = self._get_input()
            if self.reference_fn is not None:
                out = test_case._forward(module, input)
                ref_input = deepcopy(input)
                ref_module = deepcopy(module)
                expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
                test_case.assertEqual(out, expected_out, exact_dtype=False)
            if self.check_forward_only:
                return
            self.test_noncontig(test_case, module, input)
            if self.should_test_pickle:
                with tempfile.TemporaryFile() as f:
                    test_case._forward(module, input)
                    torch.save(module, f)
                    f.seek(0)
                    module_copy = torch.load(f)
                    test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))
            self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        elif isinstance(obj, tuple):
            return tuple((self.noncontiguize(o) for o in obj))
        tensor = obj
        ndim = tensor.dim()
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        assert noncontig.numel() == 1 or noncontig.numel() == 0 or (not noncontig.is_contiguous())
        noncontig.requires_grad = tensor.requires_grad
        return noncontig

    def test_noncontig(self, test_case, module, input):
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        if any((i.dim() == 0 for i in input if isinstance(i, torch.Tensor))):
            return
        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            if getattr(module, 'return_indices', False):
                output = output[0]
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])
        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            go = deepcopy(grad_output if contig_g else nc_grad_output)
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                out = test_case._forward(module, i)
                if getattr(module, 'return_indices', False):
                    out = out[0]
                grad = test_case._backward(module, i, out, go)
                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, atol=0.0001, rtol=0)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        with set_default_dtype(self.default_dtype):
            cpu_input = self._get_input()
            type_map = {torch.double: torch.float}
            cpu_input_tuple = cpu_input if isinstance(cpu_input, tuple) else (cpu_input,)
            is_any_input_complex = any((isinstance(t, torch.Tensor) and t.dtype.is_complex for t in cpu_input_tuple))
            gpu_input_tuple = to_gpu(cpu_input_tuple, type_map=type_map)
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
                gpu_p.data.copy_(cpu_p)
            test_case._zero_grad_input(cpu_input_tuple)
            test_case._zero_grad_input(gpu_input_tuple)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input_tuple)
            gpu_output = test_case._forward(gpu_module, gpu_input_tuple)
            if getattr(cpu_module, 'return_indices', False):
                cpu_output = cpu_output[0]
                gpu_output = gpu_output[0]
            test_case.assertEqual(cpu_output, gpu_output, atol=self.precision, rtol=0, exact_dtype=False)
            for _ in range(5):
                cpu_gradOutput = cpu_output.clone().normal_()
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output)
                cpu_gradInput = test_case._backward(cpu_module, cpu_input_tuple, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input_tuple, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
                for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1]):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0)
            if self.check_gradgrad and (not self.FIXME_no_cuda_gradgrad_comparison):
                cpu_output = cpu_module(*cpu_input_tuple)
                gpu_output = gpu_module(*gpu_input_tuple)
                if getattr(cpu_module, 'return_indices', False):
                    cpu_output = cpu_output[0]
                    gpu_output = gpu_output[0]
                cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
                gpu_gradOutput.requires_grad = True
                cpu_gradInputs = torch.autograd.grad(cpu_output, cpu_input_tuple + tuple(cpu_module.parameters()), cpu_gradOutput, create_graph=True)
                gpu_gradInputs = torch.autograd.grad(gpu_output, gpu_input_tuple + tuple(gpu_module.parameters()), gpu_gradOutput, create_graph=True)
                for cpu_d_i, gpu_d_i in zip(cpu_gradInputs, gpu_gradInputs):
                    test_case.assertEqual(cpu_d_i, gpu_d_i, atol=self.precision, rtol=0, exact_dtype=False)
                if is_any_input_complex:
                    outputs_cpu = cpu_output.sum().abs() + sum((x.sum().abs() for x in cpu_gradInputs))
                    outputs_gpu = gpu_output.sum().abs() + sum((x.sum().abs() for x in gpu_gradInputs))
                else:
                    outputs_cpu = cpu_output.sum() + sum((x.sum() for x in cpu_gradInputs))
                    outputs_gpu = gpu_output.sum() + sum((x.sum() for x in gpu_gradInputs))
                cpu_gg = torch.autograd.grad(outputs_cpu, cpu_input_tuple + (cpu_gradOutput,) + tuple(cpu_module.parameters()), retain_graph=True)
                gpu_gg = torch.autograd.grad(outputs_gpu, gpu_input_tuple + (gpu_gradOutput,) + tuple(gpu_module.parameters()), retain_graph=True)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
                for cpu_d_p, gpu_d_p in zip(cpu_gg, gpu_gg):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0, exact_dtype=False)
            self.test_noncontig(test_case, gpu_module, gpu_input_tuple)