from typing import TYPE_CHECKING, List, Union
def _concatenate_chunked_arrays(arrs: 'pyarrow.ChunkedArray') -> 'pyarrow.ChunkedArray':
    """
    Concatenate provided chunked arrays into a single chunked array.
    """
    from ray.data.extensions import ArrowTensorType, ArrowVariableShapedTensorType
    chunks = []
    type_ = None
    for arr in arrs:
        if type_ is None:
            type_ = arr.type
        else:
            if isinstance(type_, (ArrowTensorType, ArrowVariableShapedTensorType)):
                raise ValueError(f'_concatenate_chunked_arrays should only be used on non-tensor extension types, but got a chunked array of type {type_}.')
            assert type_ == arr.type, f'Types mismatch: {type_} != {arr.type}'
        chunks.extend(arr.chunks)
    return pyarrow.chunked_array(chunks, type=type_)