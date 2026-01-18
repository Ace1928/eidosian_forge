import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _str_intern(inp, *, tensor_contents=None):
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
        return _functorch_wrapper_str_intern(inp, tensor_contents=tensor_contents)
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter
    if inp.is_nested:
        prefix = 'nested_tensor('
    elif is_plain_tensor:
        prefix = 'tensor('
    else:
        prefix = f'{type(inp).__name__}('
    indent = len(prefix)
    suffixes = []
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)
    if self.device.type != torch._C._get_default_device() or (self.device.type == 'cuda' and torch.cuda.current_device() != self.device.index) or self.device.type == 'mps':
        suffixes.append("device='" + str(self.device) + "'")
    if self.device.type in ['xla', 'lazy', 'ipu', 'mtia']:
        self = self.to('cpu')
    _default_complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    has_default_dtype = self.dtype in (torch.get_default_dtype(), _default_complex_dtype, torch.int64, torch.bool)
    if self.is_sparse:
        suffixes.append('size=' + str(tuple(self.shape)))
        from torch._subclasses.fake_tensor import FakeTensor
        if not self.is_meta and (not isinstance(self, FakeTensor)):
            suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        if not custom_contents_provided:
            indices_prefix = 'indices=tensor('
            indices = self._indices().detach()
            indices_str = _tensor_str(indices, indent + len(indices_prefix))
            if indices.numel() == 0:
                indices_str += ', size=' + str(tuple(indices.shape))
            values_prefix = 'values=tensor('
            values = self._values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ', size=' + str(tuple(values.shape))
            tensor_str = indices_prefix + indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    elif self.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        suffixes.append('size=' + str(tuple(self.shape)))
        suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        if not custom_contents_provided:
            compressed_indices_method, plain_indices_method = {torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices), torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices)}[self.layout]
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                cdimname, pdimname = ('row', 'column')
            else:
                cdimname, pdimname = ('column', 'row')
            compressed_indices_prefix = f'c{cdimname[:3]}_indices=tensor('
            compressed_indices = compressed_indices_method(self).detach()
            compressed_indices_str = _tensor_str(compressed_indices, indent + len(compressed_indices_prefix))
            if compressed_indices.numel() == 0:
                compressed_indices_str += ', size=' + str(tuple(compressed_indices.shape))
            plain_indices_prefix = f'{pdimname[:3]}_indices=tensor('
            plain_indices = plain_indices_method(self).detach()
            plain_indices_str = _tensor_str(plain_indices, indent + len(plain_indices_prefix))
            if plain_indices.numel() == 0:
                plain_indices_str += ', size=' + str(tuple(plain_indices.shape))
            values_prefix = 'values=tensor('
            values = self.values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ', size=' + str(tuple(values.shape))
            tensor_str = compressed_indices_prefix + compressed_indices_str + '),\n' + ' ' * indent + plain_indices_prefix + plain_indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    elif self.is_quantized:
        suffixes.append('size=' + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        suffixes.append('quantization_scheme=' + str(self.qscheme()))
        if self.qscheme() == torch.per_tensor_affine or self.qscheme() == torch.per_tensor_symmetric:
            suffixes.append('scale=' + str(self.q_scale()))
            suffixes.append('zero_point=' + str(self.q_zero_point()))
        elif self.qscheme() == torch.per_channel_affine or self.qscheme() == torch.per_channel_symmetric or self.qscheme() == torch.per_channel_affine_float_qparams:
            suffixes.append('scale=' + str(self.q_per_channel_scales()))
            suffixes.append('zero_point=' + str(self.q_per_channel_zero_points()))
            suffixes.append('axis=' + str(self.q_per_channel_axis()))
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    elif self.is_nested:
        if not custom_contents_provided:

            def indented_str(s, indent):
                return '\n'.join((f'  {line}' for line in s.split('\n')))
            strs = ',\n'.join((indented_str(str(t), indent + 1) for t in torch.ops.aten.unbind.int(self, 0)))
            tensor_str = f'[\n{strs}\n]'
    elif torch._is_functional_tensor(self):
        prefix = '_to_functional_tensor('
        tensor_str = repr(torch._from_functional_tensor(self))
    else:
        from torch._subclasses.fake_tensor import FakeTensor
        if self.is_meta or isinstance(self, FakeTensor):
            suffixes.append('size=' + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                tensor_str = '...'
        elif self.numel() == 0 and (not self.is_sparse):
            if self.dim() != 1:
                suffixes.append('size=' + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                tensor_str = '[]'
        else:
            if not PRINT_OPTS.edgeitems:
                suffixes.append('size=' + str(tuple(self.shape)))
            if not has_default_dtype:
                suffixes.append('dtype=' + str(self.dtype))
            if not custom_contents_provided:
                if self.layout != torch.strided:
                    tensor_str = _tensor_str(self.to_dense(), indent)
                else:
                    tensor_str = _tensor_str(self, indent)
    if self.layout != torch.strided:
        suffixes.append('layout=' + str(self.layout))
    grad_fn_name = None
    try:
        grad_fn = inp.grad_fn
    except RuntimeError:
        grad_fn_name = 'Invalid'
    if grad_fn_name is None and grad_fn is not None:
        grad_fn_name = type(grad_fn).__name__
        if grad_fn_name == 'CppFunction':
            grad_fn_name = grad_fn.name().rsplit('::', 1)[-1]
    if grad_fn_name is not None:
        suffixes.append(f'grad_fn=<{grad_fn_name}>')
    elif inp.requires_grad:
        suffixes.append('requires_grad=True')
    if self.has_names():
        suffixes.append(f'names={self.names}')
    if tangent is not None:
        suffixes.append(f'tangent={tangent}')
    string_repr = _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse)
    if isinstance(self, torch.nn.Parameter) and (not is_plain_tensor):
        string_repr = f'Parameter({string_repr})'
    return string_repr