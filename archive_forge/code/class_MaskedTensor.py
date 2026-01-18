import warnings
import torch
from torch.overrides import get_default_nowrap_functions
class MaskedTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        if is_masked_tensor(data) or not torch.is_tensor(data):
            raise TypeError('data must be a Tensor')
        if is_masked_tensor(mask) or not torch.is_tensor(mask):
            raise TypeError('mask must be a Tensor')
        kwargs = {}
        kwargs['device'] = data.device
        kwargs['dtype'] = data.dtype
        kwargs['layout'] = data.layout
        kwargs['requires_grad'] = requires_grad
        kwargs['dispatch_sizes_strides_policy'] = 'strides'
        kwargs['dispatch_layout'] = True
        warnings.warn('The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.', UserWarning)
        if data.requires_grad:
            warnings.warn('It is not recommended to create a MaskedTensor with a tensor that requires_grad. To avoid this, you can use data.clone().detach()', UserWarning)
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

    def _preprocess_data(self, data, mask):
        from .._ops import _sparse_coo_where, _sparse_csr_where
        if data.layout != mask.layout:
            raise TypeError('data and mask must have the same layout.')
        if data.layout == torch.sparse_coo:
            data = data.coalesce()
            mask = mask.coalesce()
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, torch.tensor(0))
        elif data.layout == torch.sparse_csr:
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, torch.tensor(0))
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        data = self._masked_data
        mask = self.get_mask()
        if type(data) != type(mask):
            raise TypeError(f'data and mask must have the same type. Got {type(data)} and {type(mask)}')
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f'data layout of {data.layout} is not supported.')
        if data.layout == torch.sparse_coo:
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError('data and mask are both sparse COO tensors but do not have the same indices.')
        elif data.layout == torch.sparse_csr:
            if not _tensors_match(data.crow_indices(), mask.crow_indices(), exact=True) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError('data and mask are both sparse CSR tensors but do not share either crow or col indices.')
        if mask.dtype != torch.bool:
            raise TypeError('mask must have dtype bool.')
        if not (data.dtype == torch.float16 or data.dtype == torch.float32 or data.dtype == torch.float64 or (data.dtype == torch.bool) or (data.dtype == torch.int8) or (data.dtype == torch.int16) or (data.dtype == torch.int32) or (data.dtype == torch.int64)):
            raise TypeError(f'{data.dtype} is not supported in MaskedTensor.')
        if data.dim() != mask.dim():
            raise ValueError('data.dim() must equal mask.dim()')
        if data.size() != mask.size():
            raise ValueError('data.size() must equal mask.size()')

    def __init__(self, data, mask, requires_grad=False):
        self._preprocess_data(data, mask)
        self._validate_members()

    @staticmethod
    def _from_values(data, mask):
        """ Differentiable constructor for MaskedTensor """

        class Constructor(torch.autograd.Function):

            @staticmethod
            def forward(ctx, data, mask):
                return MaskedTensor(data, mask)

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output, None)
        result = Constructor.apply(data, mask)
        return result

    def _set_data_mask(self, data, mask):
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    def __repr__(self):
        formatter = '{0:8.4f}'
        if self.dim() == 0:
            scalar_data = self.get_data().item()
            data_formatted = formatter.format(scalar_data) if isinstance(scalar_data, float) else str(scalar_data)
            if not self.get_mask().item():
                data_formatted = '--'
            return 'MaskedTensor(' + data_formatted + ', ' + str(self.get_mask().item()) + ')'
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = '\n'.join(('  ' + si for si in s.split('\n')))
        return 'MaskedTensor(\n' + s + '\n)'

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        from ._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE
        if func in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)
        if not all((issubclass(cls, t) for t in types)):
            return NotImplemented
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        func = func.overloadpacket
        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE
        if func in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)
        msg = f'{func.__name__} is not implemented in __torch_dispatch__ for MaskedTensor.\nIf you would like this operator to be supported, please file an issue for a feature request at https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\nIn the case that the semantics for the operator are not trivial, it would be appreciated to also include a proposal for the semantics.'
        warnings.warn(msg)
        return NotImplemented

    def __lt__(self, other):
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        return MaskedTensor(self.get_data() < other, self.get_mask())

    def to_tensor(self, value):
        return self.get_data().masked_fill(~self.get_mask(), value)

    def get_data(self):

        class GetData(torch.autograd.Function):

            @staticmethod
            def forward(ctx, self):
                return self._masked_data

            @staticmethod
            def backward(ctx, grad_output):
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())
        return GetData.apply(self)

    def get_mask(self):
        return self._masked_mask

    def is_sparse_coo(self):
        return self.layout == torch.sparse_coo

    def is_sparse_csr(self):
        return self.layout == torch.sparse_csr

    @property
    def is_sparse(self):
        return self.is_sparse_coo() or self.is_sparse_csr()