import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
class BlockSparseTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, values, layout):
        kwargs = {}
        kwargs['device'] = values.device
        kwargs['dtype'] = values.dtype
        kwargs['layout'] = values.layout
        kwargs['requires_grad'] = values.requires_grad
        assert values.ndim == 4
        B, _, block_size, _ = values.shape
        C, h, w = layout.shape
        shape = (B, C, block_size * h, block_size * w)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, values, layout):
        assert values.shape[-2] == values.shape[-1]
        assert values.device == layout.device, 'Both values and layout need to reside on the same device'
        block_size = values.shape[-1]
        assert block_size >= 16, 'Minimum block size is 16, for now at least'
        self.__values = values
        self.__layout = layout
        if blocksparse_matmul:
            self._initialize_triton_ops()
        else:
            self.__sparse_dot_sdd = None
            self.__sparse_dot_dsd = None
            self.__sparse_softmax = None

    def _initialize_triton_ops(self):
        block_size = self.__values.shape[-1]
        self.__sparse_dot_sdd = blocksparse_matmul(self.__layout, block_size, 'sdd', trans_a=False, trans_b=True, device=self.__layout.device)
        self.__sparse_dot_dsd = blocksparse_matmul(self.__layout, block_size, 'dsd', trans_a=False, trans_b=False, device=self.__layout.device)
        self.__sparse_softmax = blocksparse_softmax(self.__layout, block_size, device=self.__layout.device)

    def __repr__(self):
        return f'block_sparse_tensor(shape={self.shape}, values={self.__values})'

    def values(self):
        return self.__values

    @classmethod
    def _raw_wrap(cls, values, layout, sparse_dot_sdd, sparse_dot_dsd, sparse_softmax):
        matrix = cls.__new__(cls, values, layout)
        matrix.__values = values
        matrix.__layout = layout
        matrix.__sparse_dot_sdd = sparse_dot_sdd
        matrix.__sparse_dot_dsd = sparse_dot_dsd
        matrix.__sparse_softmax = sparse_softmax
        return matrix

    @classmethod
    def _wrap(cls, values, bmat):
        matrix = cls.__new__(cls, values, bmat.__layout)
        matrix.__values = values
        matrix.__layout = bmat.__layout
        matrix.__sparse_dot_sdd = bmat.__sparse_dot_sdd
        matrix.__sparse_dot_dsd = bmat.__sparse_dot_dsd
        matrix.__sparse_softmax = bmat.__sparse_softmax
        return matrix

    @classmethod
    def _bmm(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and type(arg1) is torch.Tensor):
            return NotImplemented
        if _can_use_triton(arg1):
            res = arg0.__sparse_dot_dsd(arg0.__values, arg1)
        else:
            res = _spmm(arg1, arg0.__layout, arg0.__values)
        return res

    @classmethod
    def _masked_matmul(cls, a, b, mask):
        if not (type(a) is torch.Tensor and type(b) is torch.Tensor):
            return NotImplemented
        b = b.transpose(-2, -1)
        assert b.is_contiguous()
        if _can_use_triton(a):
            res = mask.__sparse_dot_sdd(a, b)
        else:
            res = _sddmm(a, b, mask.__layout)
        return cls._wrap(res, mask)

    @classmethod
    def _softmax(cls, arg0, dim):
        if not (dim == -1 or dim == 2):
            return NotImplemented
        if _can_use_triton(arg0):
            res = arg0.__sparse_softmax(arg0.__values)
        else:
            res = _softmax(arg0.__layout, arg0.__values)
        return cls._wrap(res, arg0)

    @classmethod
    def _to(cls, arg0, device):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        return cls(arg0.__values.to(device=device), arg0.__layout)

    @classmethod
    def _copy(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        assert arg0.shape == arg1.shape
        av0, av1 = (arg0.__values, arg1.__values)
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = (arg0.__layout, arg1.__layout)
        av0.resize_as_(av1).copy_(av1)
        out = cls(arg0.__values, arg0.__layout)
        arg0.__sparse_dot_sdd = out.__sparse_dot_sdd
        arg0.__sparse_dot_dsd = out.__sparse_dot_dsd
        arg0.__sparse_softmax = out.__sparse_softmax
        return arg0

    @classmethod
    def _equal(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        if arg0.shape != arg1.shape:
            return False
        if not torch.equal(arg0.__values, arg1.__values):
            return False
        if not torch.equal(arg0.__layout, arg1.__layout):
            return False
        return True

    @classmethod
    def _to_dense(cls, arg0):
        out = torch.zeros(arg0.shape, dtype=arg0.dtype, device=arg0.device)
        values = arg0.__values
        layout = arg0.__layout
        block_size = values.shape[-1]
        blocks_i = layout.shape[-2]
        blocks_j = layout.shape[-1]
        out_r = out.reshape(arg0.shape[0], arg0.shape[1], blocks_i, block_size, blocks_j, block_size)
        for idx, (h, i, j) in enumerate(zip(*layout.nonzero(as_tuple=True))):
            out_r[:, h, i, :, j, :] = values[:, idx, :, :]
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in [torch.Tensor.bmm, torch.bmm, torch.Tensor.__matmul__, torch.matmul, torch.Tensor.matmul]:
            assert len(args) == 2
            return cls._bmm(args[0], args[1])
        if func in [torch.Tensor.softmax, torch.nn.functional.softmax, torch.softmax]:
            return cls._softmax(args[0], kwargs['dim'])
        if func == masked_matmul:
            assert len(args) == 3
            return cls._masked_matmul(args[0], args[1], args[2])
        if func in [torch.nn.functional.dropout, torch.dropout, torch.dropout_]:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)
        if func == torch.Tensor.to:
            assert len(args) >= 2
            return cls._to(args[0], args[1])
        if func in [torch.Tensor.copy_]:
            assert len(args) == 2
            return cls._copy(args[0], args[1])
        if func in [torch.Tensor.equal, torch.equal]:
            assert len(args) == 2
            return cls._equal(args[0], args[1])
        if func == torch.Tensor.to_dense:
            assert len(args) == 1
            return cls._to_dense(args[0])
        if func == torch.Tensor.detach:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)
        if func == torch.Tensor.__deepcopy__:
            x = args[0]
            memo = args[1]
            return cls._raw_wrap(x.__values.__deepcopy__(memo), x.__layout.__deepcopy__(memo), x.__sparse_dot_sdd, x.__sparse_dot_dsd, x.__sparse_softmax)
        if func in [torch.Tensor.grad.__get__, torch.Tensor._grad.__get__]:
            assert len(args) == 1
            assert len(kwargs) == 0
            x = args[0]
            return cls._wrap(x.__values.grad, x)
        if func == torch.Tensor.requires_grad_:
            func(args[0].__values)
        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            if func in torch.overrides.get_default_nowrap_functions():
                return ret
            return torch._tensor._convert(ret, cls)
        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return NotImplemented