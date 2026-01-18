from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
class TritonLangProxy:
    _memory_map: MemoryMap
    _context: ExecutionContext

    def __init__(self, memory_map: MemoryMap, context: ExecutionContext):
        self._memory_map = memory_map
        self._context = context

    @_tensor_operation
    def load(self, pointer: torch.Tensor, mask: torch.Tensor=None, other=0.0, cache_modifier='', eviction_policy='', volatile=False):
        return self._memory_map.load(pointer, mask, other)

    @_tensor_operation
    def store(self, pointer: torch.Tensor, value: torch.Tensor, mask=None):
        return self._memory_map.store(pointer, value, mask)

    @_tensor_operation
    def program_id(self, axis):
        assert axis < len(self._context.program_id)
        return torch.tensor([self._context.program_id[axis]], dtype=torch.int32, device='cuda')

    @_tensor_operation
    def num_programs(self, axis):
        assert axis < len(self._context.program_size)
        return torch.tensor([self._context.program_size[axis]], dtype=torch.int32, device='cuda')

    @_tensor_operation
    def arange(self, start, end):
        return torch.arange(start=start, end=end, dtype=torch.int32, device='cuda')

    @_tensor_operation
    def zeros(self, shape, dtype):
        for i, d in enumerate(shape):
            if not isinstance(d, debugger_constexpr):
                raise TypeError(f'Shape element {i} must have type `constexpr`')
            if not isinstance(d.value, int):
                raise TypeError(f'Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]')
        shape = [x.value for x in shape]
        if isinstance(dtype, lcore.dtype):
            if dtype.is_fp32():
                dtype = torch.float32
            elif dtype.is_fp16():
                dtype = torch.float16
            elif dtype.is_bf16():
                dtype = torch.bfloat16
            elif dtype.is_int32():
                dtype = torch.int32
            elif dtype.is_int16():
                dtype = torch.int16
            elif dtype.is_int8():
                dtype = torch.int8
            else:
                raise TypeError(f'Unsupported dtype {dtype}')
        return torch.zeros(size=shape, dtype=dtype, device='cuda')

    @_tensor_operation
    def dequantize(self, input, scale, shift, nbit, dst_ty=None):
        if dst_ty is None:
            dst_ty = torch.float16
        raise NotImplementedError()

    @_tensor_operation
    def broadcast(self, input, other):
        raise NotImplementedError()

    @_tensor_operation
    def broadcast_to(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def cat(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def reshape(self, input, shape):
        raise NotImplementedError()

    @_tensor_operation
    def dot(self, input, other, trans_a=False, trans_b=False, allow_tf32=True):
        assert input.dtype == other.dtype
        if trans_a:
            input = input.T
        if trans_b:
            other = other.T
        return torch.matmul(input=input, other=other)

    @_tensor_operation
    def atomic_cas(self, pointer, cmp, val):
        stored = self._memory_map.load(pointer, None, 0.0)
        if not isinstance(cmp, torch.Tensor):
            cmp = torch.tensor([cmp], dtype=stored.dtype, device='cuda')
        if not isinstance(val, torch.Tensor):
            val = torch.tensor([val], dtype=stored.dtype, device='cuda')
        if stored == cmp:
            self._memory_map.store(pointer, val, None)
        return stored

    @_tensor_operation
    def atomic_xchg(self, pointer, val, mask=None):
        if isinstance(val, int):
            val = torch.tensor([val], dtype=torch.int32, device='cuda')
        stored = self._memory_map.load(pointer, mask, 0.0)
        self._memory_map.store(pointer, val, mask)
        return stored

    @_tensor_operation
    def atomic_add(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = stored + val
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_max(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = torch.maximum(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_min(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0.0)
        result = torch.minimum(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_and(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_and(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_or(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_or(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def atomic_xor(self, pointer, val, mask=None):
        stored = self._memory_map.load(pointer, mask, 0)
        result = torch.bitwise_xor(stored, val)
        self._memory_map.store(pointer, result, mask)
        return stored

    @_tensor_operation
    def where(self, condition, x, y):
        condition = _primitive_to_tensor(condition)
        x = _primitive_to_tensor(x)
        y = _primitive_to_tensor(y)
        return torch.where(condition, x, y)

    @_tensor_operation
    def umulhi(self, x, y):
        raise NotImplementedError()

    @_tensor_operation
    def fdiv(self, x, y, ieee_rounding=False):
        raise NotImplementedError()

    @_tensor_operation
    def exp(self, x):
        return torch.exp(x)

    @_tensor_operation
    def log(self, x):
        return torch.log(x)

    @_tensor_operation
    def cos(self, x):
        return torch.cos(x)

    @_tensor_operation
    def sin(self, x):
        return torch.sin(x)

    @_tensor_operation
    def sqrt(self, x):
        return torch.sqrt(x)

    @_tensor_operation
    def globaltimer(self):
        raise NotImplementedError()

    @_tensor_operation
    def clock(self):
        raise NotImplementedError()

    @_tensor_operation
    def debug_barrier(self):
        raise NotImplementedError()

    @_tensor_operation
    def multiple_of(self, input, values):
        return input

    @_tensor_operation
    def max_contiguous(self, input, values):
        return input

    @_tensor_operation
    def max_constancy(self, input, values):
        return input

    @_tensor_operation
    def abs(self, x):
        return torch.abs(x)

    @_tensor_operation
    def cdiv(self, x, div):
        return (x + div - 1) // div

    @_tensor_operation
    def minimum(self, x, y):
        if isinstance(x, int):
            x = torch.tensor(x, device='cuda')
        if isinstance(y, int):
            y = torch.tensor(y, device='cuda')
        return torch.minimum(x, y)

    @_tensor_operation
    def maximum(self, x, y):
        return torch.maximum(x, y)

    @_tensor_operation
    def sigmoid(self, x):
        raise NotImplementedError()

    @_tensor_operation
    def softmax(self, x, ieee_rounding=False):
        raise NotImplementedError()

    @_tensor_operation
    def ravel(self, x):
        raise NotImplementedError()

    @_tensor_operation
    def swizzle2d(self, i, j, size_i, size_j, size_g):
        raise NotImplementedError()

    @_tensor_operation
    def zeros_like(self, input):
        raise NotImplementedError()

    @_tensor_operation
    def max(self, input, axis=None):
        if axis is None:
            return torch.max(input)
        return torch.max(input, dim=axis).values

    @_tensor_operation
    def argmax(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def min(self, input, axis=None):
        if axis is None:
            return torch.min(input)
        return torch.min(input, dim=axis).values

    @_tensor_operation
    def argmin(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def sum(self, input, axis=None):
        if axis is None:
            return torch.sum(input)
        return torch.sum(input, dim=axis)

    @_tensor_operation
    def xor_sum(self, input, axis):
        raise NotImplementedError()

    @_tensor_operation
    def cumsum(self, input, axis=None):
        if axis is None:
            return torch.cumsum(input)
        return torch.cumsum(input, dim=axis)

    @_tensor_operation
    def cumprod(self, input, axis=None):
        if axis is None:
            return torch.cumprod(input)
        return torch.cumprod(input, dim=axis)