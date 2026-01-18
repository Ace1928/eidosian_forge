from __future__ import annotations
import functools
from typing import Callable, Dict, List, Sequence, Tuple, Union
import torch
from functorch._C import dim as _C
from ._parsing import (
@functools.lru_cache(256)
def _create_rearrange_callable(tensor_ndim: int, pattern: str, **axes_lengths: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Translate an `einops`-style pattern into a callable that performs the rearrange using first-class dimensions.

    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and
    specified axes lengths, this function can be memoized.

    Args:
        tensor_ndim (int): the number of dimensions in the tensor to rearrange
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: a callable that performs the rearrangement
    """
    left, right = parse_pattern(pattern, axes_lengths)
    validate_rearrange_expressions(left, right, axes_lengths)
    n_anon_dims = sum((not dim for dim in left.composition))
    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_named_dims = len(left.identifiers) - 1
        if (pattern_ndim := (n_anon_dims + n_named_dims)) > tensor_ndim:
            raise ValueError(f'Number of dimensions in pattern ({pattern_ndim}) must be less than or equal to the number of dimensions in the tensor ({tensor_ndim})')
    else:
        n_ellipsis_dims = 0
        n_named_dims = len(left.identifiers)
        if (pattern_ndim := len(left.composition)) != tensor_ndim:
            raise ValueError(f'Number of dimensions in pattern ({pattern_ndim}) must be equal to the number of dimensions in the tensor ({tensor_ndim})')
    n_dims = n_named_dims + n_ellipsis_dims + n_anon_dims
    if n_dims == 0:
        return lambda tensor: tensor
    first_class_dims: Tuple[str, ...] = tuple((f'd{i}' for i in range(n_dims)))
    identifier_dim_map: Dict[Union[str, AnonymousAxis], Tuple[str, ...]] = {}
    anon_axes: List[AnonymousAxis] = []
    dims_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                assert isinstance(identifier, str)
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
            if not dimension:
                anon_axis = AnonymousAxis('1')
                identifier_dim_map[anon_axis] = (first_class_dims[dims_i],)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dims_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_dim_map[identifier] = tuple((first_class_dims[dims_i + j] for j in range(n_ellipsis_dims)))
            dims_i += n_ellipsis_dims
        else:
            raise ValueError(f'Unexpected dimension: {dimension}')

    def composition_to_dims(composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]) -> List[Union[str, Tuple[str, ...]]]:
        """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first
        class dims."""
        dim_composition: List[Union[str, Tuple[str, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                dim_composition.append(tuple((dim for identifier in dimension for dim in identifier_dim_map[identifier])))
            elif dimension == _ellipsis:
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                raise ValueError(f'Unexpected dimension: {dimension}')
        return dim_composition
    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    anon_dims = tuple((identifier_dim_map[axis][0] for axis in anon_axes))
    specified_lengths = tuple(((identifier_dim_map[axis][0], length) for axis, length in axes_lengths.items()))
    custom_rearrange_callable_name = 'do_rearrange'
    custom_rearrange_callable_code = f'def {custom_rearrange_callable_name}(tensor):\n    {comma_separate(first_class_dims)} = dims({n_dims})\n' + (''.join((f'    {dim}.size = {length}\n' for dim, length in specified_lengths)) if specified_lengths else '') + f'    tensor = tensor[{comma_separate(left_dims)}].order({comma_separate(right_dims)})\n' + (f'    return tensor.sum({comma_separate([anon_dims])}, keepdim=False)\n' if anon_dims else '    return tensor\n')
    exec(custom_rearrange_callable_code)
    return locals()[custom_rearrange_callable_name]