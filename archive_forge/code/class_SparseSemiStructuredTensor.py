import warnings
from collections import namedtuple
from typing import Any, Optional
import torch
class SparseSemiStructuredTensor(torch.Tensor):
    """This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    Currently, this class supports 2:4 sparsity for int8, float16 and bfloat16 dtypes.
    We also support 1:2 sparsity for float32 dtype.

    This subclass stores the dense tensor in a compressed form by only storing the specified elements and corresponding metadata.

    The subclass supports two backend, either CUTLASS or cuSPASRELt.

    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:

    compressed tensor = [ specified elements of original tensor | metadata ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    For CUTLASS backend, elements of original tensor and metadata are kept in separate tensors.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_linear
    and sparse_semi_structured_from_dense for conversion to the compressed format.

    When PyTorch is compiled with cuSPARSELt support, this subclass will call into _cslt_sparse_mm for sparse mm and
    _cslt_compress to convert into the compressed format.
    """
    _FUSE_TRANSPOSE = False
    _FORCE_CUTLASS = True
    _PROTOTYPE_WARNING_SHOWN = False

    @staticmethod
    def __new__(cls, original_tensor: Optional[torch.Tensor], original_shape: Optional[torch.Size]=None, compressed_tensor_cusparselt: Optional[torch.Tensor]=None, sparse_tensor_cutlass: Optional[torch.Tensor]=None, meta_tensor_cutlass: Optional[torch.Tensor]=None, transposed: bool=False):
        """
        Create a new instance of the class.

        When original_tensor is passed in, we compress it and store the compresed representation.
        We can also create new instance of the class from the compressed representation without the original tensor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            compressed_tensor_cusparselt: For cuSPARSELt backend, a flattened tensor to store the specified elements and metadata.
            sparse_tensor_cutlass: For CUTLASS backend, tensor to store the speficied elements.
            meta_tensor_cutlass: For CUTLASS backend, tensor to store metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.

        """
        assert compressed_tensor_cusparselt is None or (sparse_tensor_cutlass is None and meta_tensor_cutlass is None)
        if not cls._PROTOTYPE_WARNING_SHOWN:
            warnings.warn('The PyTorch API of SparseSemiStructuredTensor is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.sparse module for further information about the project.', UserWarning)
            cls._PROTOTYPE_WARNING_SHOWN = True
        if original_tensor is not None:
            previous_tensor = original_tensor
            original_shape = original_tensor.shape
        elif compressed_tensor_cusparselt is not None:
            previous_tensor = compressed_tensor_cusparselt
        elif sparse_tensor_cutlass is not None:
            previous_tensor = sparse_tensor_cutlass
        else:
            raise ValueError('All of the tensor arguments are None!')
        kwargs = {}
        kwargs['device'] = previous_tensor.device
        kwargs['dtype'] = previous_tensor.dtype
        kwargs['layout'] = previous_tensor.layout
        kwargs['requires_grad'] = False
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)

    @staticmethod
    def __get_indices_dtype(values_dtype):
        if values_dtype == torch.int8:
            return torch.int32
        elif values_dtype in (torch.float16, torch.bfloat16):
            return torch.int16
        else:
            raise RuntimeError(f'Datatype {values_dtype}  is not supported!')
        return None

    def __init__(self, original_tensor: Optional[torch.Tensor], original_shape: Optional[torch.Size]=None, compressed_tensor_cusparselt: Optional[torch.Tensor]=None, sparse_tensor_cutlass: Optional[torch.Tensor]=None, meta_tensor_cutlass: Optional[torch.Tensor]=None, transposed: bool=False) -> None:
        """SparseSemiStructuredTensor constructor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            compressed_tensor_cusparselt: For cuSPARSELt backend, a flattened tensor to store the specified elements and metadata.
            sparse_tensor_cutlass: For CUTLASS backend, tensor to store the speficied elements.
            meta_tensor_cutlass: For CUTLASS backend, tensor to store metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            None

        Raises:
            RuntimeError: If original_tensor is not a supported dtype, dim, shape, or device.
        """
        if original_tensor is not None:
            if not original_tensor.is_cuda:
                raise RuntimeError(f'Error original_tensor.device= {original_tensor.device} is not supported! Only CUDA tensors are currently supported.')
            if original_tensor.dim() != 2:
                raise RuntimeError(f'Error original_tensor.dim = {original_tensor.dim()} is not supported! Only 2d tensors are currently supported.')
            if original_tensor.dtype not in _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG:
                raise RuntimeError(f'Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! dtype must be one of: {{_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG}}')
            m, n = original_tensor.shape
            min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_rows
            min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_cols
            if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
                raise RuntimeError(f'Error original_tensor.shape {original_tensor.shape} is not supported! Both dimensions must be larger or equal than and a multiple of ({min_rows}, {min_cols})')
            compressed_tensor_cusparselt = None
            sparse_tensor_cutlass = None
            meta_tensor_cutlass = None
            if self._FORCE_CUTLASS:
                from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
                sparse_tensor_cutlass, meta_tensor_cutlass = sparse_semi_structured_from_dense_cutlass(original_tensor)
            else:
                compressed_tensor_cusparselt = torch._cslt_compress(original_tensor)
        self.original_tensor = None
        self.compressed_tensor_cusparselt = compressed_tensor_cusparselt
        self.sparse_tensor_cutlass = sparse_tensor_cutlass
        self.meta_tensor_cutlass = meta_tensor_cutlass
        self.transposed = transposed
        self.original_shape = original_shape

    def __tensor_flatten__(self):
        if self.compressed_tensor_cusparselt is not None:
            return (['compressed_tensor_cusparselt'], (self.original_shape, self.transposed))
        else:
            return (['sparse_tensor_cutlass', 'meta_tensor_cutlass'], (self.original_shape, self.transposed))

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        original_shape, transposed = meta
        if len(inner_tensors) == 2:
            sparse_tensor_cutlass = inner_tensors['sparse_tensor_cutlass']
            meta_tensor_cutlass = inner_tensors['meta_tensor_cutlass']
            compressed_tensor_cusparselt = None
        elif len(inner_tensors) == 1:
            sparse_tensor_cutlass = None
            meta_tensor_cutlass = None
            compressed_tensor_cusparselt = inner_tensors['compressed_tensor_cusparselt']
        else:
            raise RuntimeError(f'Expected 1 or 2 inner tensors but got {len(inner_tensors)}')
        return SparseSemiStructuredTensor(None, original_shape=original_shape, compressed_tensor_cusparselt=compressed_tensor_cusparselt, sparse_tensor_cutlass=sparse_tensor_cutlass, meta_tensor_cutlass=meta_tensor_cutlass, transposed=transposed)

    def __repr__(self) -> str:
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        return f'SparseSemiStructuredTensor(shape={self.shape}, transposed={self.transposed}values={self.values()}metadata={self.indices()})'
    __torch_function__ = torch._C._disabled_torch_function_impl

    def _pad_tensor_for_matmul(self, original_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        assert original_tensor.dim() == 2
        m, n = original_tensor.shape
        min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_rows
        min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_cols
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(original_tensor, (0, to_pad_n, 0, to_pad_m))
        else:
            return original_tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        """Overload __torch_dispatch__ to use torch._sparse_semi_structured_linear.

        `torch.structured_sparse_linear` uses accelerated sparse CUTLASS kernels.
        In the future we plan to also add in support for cuSPARSELt kernels.

        Args:
            func: The function being dispatched.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation.

        Raises:
            NotImplementedError: If the dispatched operation is not implemented.
        """
        if func is torch.ops.aten.detach.default:
            return SparseSemiStructuredTensor(args[0].original_tensor, original_shape=args[0].shape, compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt, sparse_tensor_cutlass=args[0].sparse_tensor_cutlass, meta_tensor_cutlass=args[0].meta_tensor_cutlass, transposed=args[0].transposed)
        if func is torch.ops.aten.t.default:
            return SparseSemiStructuredTensor(args[0].original_tensor, original_shape=torch.Size([args[0].shape[1], args[0].shape[0]]), compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt, sparse_tensor_cutlass=args[0].sparse_tensor_cutlass, meta_tensor_cutlass=args[0].meta_tensor_cutlass, transposed=not args[0].transposed)
        if func is torch.ops.aten.addmm.default:
            bias, input_A, input_B = args
            if isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = input_B._pad_tensor_for_matmul(input_A)
                if input_B.compressed_tensor_cusparselt is None:
                    assert input_B.sparse_tensor_cutlass is not None and input_B.meta_tensor_cutlass is not None
                    res = torch._sparse_semi_structured_linear(input_A_padded, input_B.sparse_tensor_cutlass, input_B.meta_tensor_cutlass, bias=bias)
                else:
                    res = torch._cslt_sparse_mm(input_B.compressed_tensor_cusparselt, input_A_padded.t(), bias=bias).t()
                return res[:row, :]
        if func is torch.ops.aten.mm.default:
            input_A, input_B = args
            if isinstance(input_A, cls) and (not input_A.transposed):
                row, col = input_B.shape
                input_B_padded = input_A._pad_tensor_for_matmul(input_B)
                if input_A.compressed_tensor_cusparselt is None:
                    assert input_A.sparse_tensor_cutlass is not None and input_A.meta_tensor_cutlass is not None
                    res = torch._sparse_semi_structured_linear(input_B_padded.t(), input_A.sparse_tensor_cutlass, input_A.meta_tensor_cutlass).t()
                else:
                    res = torch._cslt_sparse_mm(input_A.compressed_tensor_cusparselt, input_B_padded, bias=None)
                return res[:, :col]
            elif isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = input_B._pad_tensor_for_matmul(input_A)
                if input_B.compressed_tensor_cusparselt is None:
                    assert input_B.sparse_tensor_cutlass is not None and input_B.meta_tensor_cutlass is not None
                    res = torch._sparse_semi_structured_linear(input_A_padded, input_B.sparse_tensor_cutlass, input_B.meta_tensor_cutlass)
                else:
                    res = torch._cslt_sparse_mm(input_B.compressed_tensor_cusparselt, input_A_padded.t(), bias=None).t()
                return res[:row, :]
        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            shape = input_tensor.shape
            input_tensor_2d = input_tensor.view(-1, shape[-1])
            row, col = input_tensor_2d.shape
            input_tensor_2d_padded = weight._pad_tensor_for_matmul(input_tensor_2d)
            if isinstance(weight, cls):
                if weight.compressed_tensor_cusparselt is None:
                    assert weight.sparse_tensor_cutlass is not None and weight.meta_tensor_cutlass is not None
                    res = torch._sparse_semi_structured_linear(input_tensor_2d_padded, weight.sparse_tensor_cutlass, weight.meta_tensor_cutlass, bias=bias)
                else:
                    res = torch._cslt_sparse_mm(weight.compressed_tensor_cusparselt, input_tensor_2d_padded.t(), bias=bias).t()
                return res[:row, :].view(*shape[:-1], -1)
        if func is torch.ops.aten.values.default:
            if args[0].compressed_tensor_cusparselt is None:
                return args[0].sparse_tensor_cutlass.detach()
            else:
                m, k = args[0].shape
                num_kept_elements = m * k // 2
                return args[0].compressed_tensor_cusparselt[:num_kept_elements].view(m, k // 2)
        if func is torch.ops.aten.indices.default:
            if args[0].compressed_tensor_cusparselt is None:
                return args[0].meta_tensor_cutlass
            else:
                m, k = args[0].shape
                num_kept_elements = m * k // 2
                metadata = args[0].compressed_tensor_cusparselt[num_kept_elements:].view(m, -1)
                indices_dtype = SparseSemiStructuredTensor.__get_indices_dtype(args[0].dtype)
                return metadata.view(indices_dtype)
        error_string = '\n'.join([f'func {func} with args: '] + [f'arg{i}: {arg}' for i, arg in enumerate(args)])
        raise NotImplementedError(error_string)

    def to_dense(self):
        if self.compressed_tensor_cusparselt is not None:
            raise RuntimeError('Converting to dense is not yet supported by cuSPARSELt backend!')
        from torch.sparse._semi_structured_conversions import sparse_semi_structured_to_dense_cutlass
        return sparse_semi_structured_to_dense_cutlass(self.sparse_tensor_cutlass, self.meta_tensor_cutlass)