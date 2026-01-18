import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
class TensorLikePair(Pair):
    """Pair for :class:`torch.Tensor`-like inputs.

    Kwargs:
        allow_subclasses (bool):
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the type are selected. See :func:assert_close: for details.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the type are selected. See :func:assert_close: for details.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same ``layout``. If this
            check is disabled, tensors with different ``layout``'s are converted to strided tensors before being
            compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
    """

    def __init__(self, actual: Any, expected: Any, *, id: Tuple[Any, ...]=(), allow_subclasses: bool=True, rtol: Optional[float]=None, atol: Optional[float]=None, equal_nan: bool=False, check_device: bool=True, check_dtype: bool=True, check_layout: bool=True, check_stride: bool=False, **other_parameters: Any):
        actual, expected = self._process_inputs(actual, expected, id=id, allow_subclasses=allow_subclasses)
        super().__init__(actual, expected, id=id, **other_parameters)
        self.rtol, self.atol = get_tolerances(actual, expected, rtol=rtol, atol=atol, id=self.id)
        self.equal_nan = equal_nan
        self.check_device = check_device
        self.check_dtype = check_dtype
        self.check_layout = check_layout
        self.check_stride = check_stride

    def _process_inputs(self, actual: Any, expected: Any, *, id: Tuple[Any, ...], allow_subclasses: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        directly_related = isinstance(actual, type(expected)) or isinstance(expected, type(actual))
        if not directly_related:
            self._inputs_not_supported()
        if not allow_subclasses and type(actual) is not type(expected):
            self._inputs_not_supported()
        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        return (actual, expected)

    def _to_tensor(self, tensor_like: Any) -> torch.Tensor:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like
        try:
            return torch.as_tensor(tensor_like)
        except Exception:
            self._inputs_not_supported()

    def _check_supported(self, tensor: torch.Tensor, *, id: Tuple[Any, ...]) -> None:
        if tensor.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
            raise ErrorMeta(ValueError, f'Unsupported tensor layout {tensor.layout}', id=id)

    def compare(self) -> None:
        actual, expected = (self.actual, self.expected)
        self._compare_attributes(actual, expected)
        if any((input.device.type == 'meta' for input in (actual, expected))):
            return
        actual, expected = self._equalize_attributes(actual, expected)
        self._compare_values(actual, expected)

    def _compare_attributes(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        """Checks if the attributes of two tensors match.

        Always checks

        - the :attr:`~torch.Tensor.shape`,
        - whether both inputs are quantized or not,
        - and if they use the same quantization scheme.

        Checks for

        - :attr:`~torch.Tensor.layout`,
        - :meth:`~torch.Tensor.stride`,
        - :attr:`~torch.Tensor.device`, and
        - :attr:`~torch.Tensor.dtype`

        are optional and can be disabled through the corresponding ``check_*`` flag during construction of the pair.
        """

        def raise_mismatch_error(attribute_name: str, actual_value: Any, expected_value: Any) -> NoReturn:
            self._fail(AssertionError, f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.")
        if actual.shape != expected.shape:
            raise_mismatch_error('shape', actual.shape, expected.shape)
        if actual.is_quantized != expected.is_quantized:
            raise_mismatch_error('is_quantized', actual.is_quantized, expected.is_quantized)
        elif actual.is_quantized and actual.qscheme() != expected.qscheme():
            raise_mismatch_error('qscheme()', actual.qscheme(), expected.qscheme())
        if actual.layout != expected.layout:
            if self.check_layout:
                raise_mismatch_error('layout', actual.layout, expected.layout)
        elif actual.layout == torch.strided and self.check_stride and (actual.stride() != expected.stride()):
            raise_mismatch_error('stride()', actual.stride(), expected.stride())
        if self.check_device and actual.device != expected.device:
            raise_mismatch_error('device', actual.device, expected.device)
        if self.check_dtype and actual.dtype != expected.dtype:
            raise_mismatch_error('dtype', actual.dtype, expected.dtype)

    def _equalize_attributes(self, actual: torch.Tensor, expected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equalizes some attributes of two tensors for value comparison.

        If ``actual`` and ``expected`` are ...

        - ... not on the same :attr:`~torch.Tensor.device`, they are moved CPU memory.
        - ... not of the same ``dtype``, they are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`).
        - ... not of the same ``layout``, they are converted to strided tensors.

        Args:
            actual (Tensor): Actual tensor.
            expected (Tensor): Expected tensor.

        Returns:
            (Tuple[Tensor, Tensor]): Equalized tensors.
        """
        if actual.is_mps or expected.is_mps:
            actual = actual.cpu()
            expected = expected.cpu()
        if actual.device != expected.device:
            actual = actual.cpu()
            expected = expected.cpu()
        if actual.dtype != expected.dtype:
            dtype = torch.promote_types(actual.dtype, expected.dtype)
            actual = actual.to(dtype)
            expected = expected.to(dtype)
        if actual.layout != expected.layout:
            actual = actual.to_dense() if actual.layout != torch.strided else actual
            expected = expected.to_dense() if expected.layout != torch.strided else expected
        return (actual, expected)

    def _compare_values(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        if actual.is_quantized:
            compare_fn = self._compare_quantized_values
        elif actual.is_sparse:
            compare_fn = self._compare_sparse_coo_values
        elif actual.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
            compare_fn = self._compare_sparse_compressed_values
        else:
            compare_fn = self._compare_regular_values_close
        compare_fn(actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan)

    def _compare_quantized_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
        """Compares quantized tensors by comparing the :meth:`~torch.Tensor.dequantize`'d variants for closeness.

        .. note::

            A detailed discussion about why only the dequantized variant is checked for closeness rather than checking
            the individual quantization parameters for closeness and the integer representation for equality can be
            found in https://github.com/pytorch/pytorch/issues/68548.
        """
        return self._compare_regular_values_close(actual.dequantize(), expected.dequantize(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier=lambda default_identifier: f'Quantized {default_identifier.lower()}')

    def _compare_sparse_coo_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
        """Compares sparse COO tensors by comparing

        - the number of sparse dimensions,
        - the number of non-zero elements (nnz) for equality,
        - the indices for equality, and
        - the values for closeness.
        """
        if actual.sparse_dim() != expected.sparse_dim():
            self._fail(AssertionError, f'The number of sparse dimensions in sparse COO tensors does not match: {actual.sparse_dim()} != {expected.sparse_dim()}')
        if actual._nnz() != expected._nnz():
            self._fail(AssertionError, f'The number of specified values in sparse COO tensors does not match: {actual._nnz()} != {expected._nnz()}')
        self._compare_regular_values_equal(actual._indices(), expected._indices(), identifier='Sparse COO indices')
        self._compare_regular_values_close(actual._values(), expected._values(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier='Sparse COO values')

    def _compare_sparse_compressed_values(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool) -> None:
        """Compares sparse compressed tensors by comparing

        - the number of non-zero elements (nnz) for equality,
        - the plain indices for equality,
        - the compressed indices for equality, and
        - the values for closeness.
        """
        format_name, compressed_indices_method, plain_indices_method = {torch.sparse_csr: ('CSR', torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_csc: ('CSC', torch.Tensor.ccol_indices, torch.Tensor.row_indices), torch.sparse_bsr: ('BSR', torch.Tensor.crow_indices, torch.Tensor.col_indices), torch.sparse_bsc: ('BSC', torch.Tensor.ccol_indices, torch.Tensor.row_indices)}[actual.layout]
        if actual._nnz() != expected._nnz():
            self._fail(AssertionError, f'The number of specified values in sparse {format_name} tensors does not match: {actual._nnz()} != {expected._nnz()}')
        actual_compressed_indices = compressed_indices_method(actual)
        expected_compressed_indices = compressed_indices_method(expected)
        indices_dtype = torch.promote_types(actual_compressed_indices.dtype, expected_compressed_indices.dtype)
        self._compare_regular_values_equal(actual_compressed_indices.to(indices_dtype), expected_compressed_indices.to(indices_dtype), identifier=f'Sparse {format_name} {compressed_indices_method.__name__}')
        self._compare_regular_values_equal(plain_indices_method(actual).to(indices_dtype), plain_indices_method(expected).to(indices_dtype), identifier=f'Sparse {format_name} {plain_indices_method.__name__}')
        self._compare_regular_values_close(actual.values(), expected.values(), rtol=rtol, atol=atol, equal_nan=equal_nan, identifier=f'Sparse {format_name} values')

    def _compare_regular_values_equal(self, actual: torch.Tensor, expected: torch.Tensor, *, equal_nan: bool=False, identifier: Optional[Union[str, Callable[[str], str]]]=None) -> None:
        """Checks if the values of two tensors are equal."""
        self._compare_regular_values_close(actual, expected, rtol=0, atol=0, equal_nan=equal_nan, identifier=identifier)

    def _compare_regular_values_close(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float, equal_nan: bool, identifier: Optional[Union[str, Callable[[str], str]]]=None) -> None:
        """Checks if the values of two tensors are close up to a desired tolerance."""
        matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if torch.all(matches):
            return
        if actual.shape == torch.Size([]):
            msg = make_scalar_mismatch_msg(actual.item(), expected.item(), rtol=rtol, atol=atol, identifier=identifier)
        else:
            msg = make_tensor_mismatch_msg(actual, expected, matches, rtol=rtol, atol=atol, identifier=identifier)
        self._fail(AssertionError, msg)

    def extra_repr(self) -> Sequence[str]:
        return ('rtol', 'atol', 'equal_nan', 'check_device', 'check_dtype', 'check_layout', 'check_stride')