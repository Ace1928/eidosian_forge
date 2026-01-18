import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
class TypedSequence:
    """
    This data container generalizes the typing when instantiating pyarrow arrays, tables or batches.

    More specifically it adds several features:
    - Support extension types like ``datasets.features.Array2DExtensionType``:
        By default pyarrow arrays don't return extension arrays. One has to call
        ``pa.ExtensionArray.from_storage(type, pa.array(data, type.storage_type))``
        in order to get an extension array.
    - Support for ``try_type`` parameter that can be used instead of ``type``:
        When an array is transformed, we like to keep the same type as before if possible.
        For example when calling :func:`datasets.Dataset.map`, we don't want to change the type
        of each column by default.
    - Better error message when a pyarrow array overflows.

    Example::

        from datasets.features import Array2D, Array2DExtensionType, Value
        from datasets.arrow_writer import TypedSequence
        import pyarrow as pa

        arr = pa.array(TypedSequence([1, 2, 3], type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence([1, 2, 3], try_type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence(["foo", "bar"], try_type=Value("int32")))
        assert arr.type == pa.string()

        arr = pa.array(TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64")))
        assert arr.type == Array2DExtensionType((1, 3), "int64")

        table = pa.Table.from_pydict({
            "image": TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64"))
        })
        assert table["image"].type == Array2DExtensionType((1, 3), "int64")

    """

    def __init__(self, data: Iterable, type: Optional[FeatureType]=None, try_type: Optional[FeatureType]=None, optimized_int_type: Optional[FeatureType]=None):
        if type is not None and try_type is not None:
            raise ValueError('You cannot specify both type and try_type')
        self.data = data
        self.type = type
        self.try_type = try_type
        self.optimized_int_type = optimized_int_type
        self.trying_type = self.try_type is not None
        self.trying_int_optimization = optimized_int_type is not None and type is None and (try_type is None)
        self._inferred_type = None

    def get_inferred_type(self) -> FeatureType:
        """Return the inferred feature type.
        This is done by converting the sequence to an Arrow array, and getting the corresponding
        feature type.

        Since building the Arrow array can be expensive, the value of the inferred type is cached
        as soon as pa.array is called on the typed sequence.

        Returns:
            FeatureType: inferred feature type of the sequence.
        """
        if self._inferred_type is None:
            self._inferred_type = generate_from_arrow_type(pa.array(self).type)
        return self._inferred_type

    @staticmethod
    def _infer_custom_type_and_encode(data: Iterable) -> Tuple[Iterable, Optional[FeatureType]]:
        """Implement type inference for custom objects like PIL.Image.Image -> Image type.

        This function is only used for custom python objects that can't be direclty passed to build
        an Arrow array. In such cases is infers the feature type to use, and it encodes the data so
        that they can be passed to an Arrow array.

        Args:
            data (Iterable): array of data to infer the type, e.g. a list of PIL images.

        Returns:
            Tuple[Iterable, Optional[FeatureType]]: a tuple with:
                - the (possibly encoded) array, if the inferred feature type requires encoding
                - the inferred feature type if the array is made of supported custom objects like
                    PIL images, else None.
        """
        if config.PIL_AVAILABLE and 'PIL' in sys.modules:
            import PIL.Image
            non_null_idx, non_null_value = first_non_null_value(data)
            if isinstance(non_null_value, PIL.Image.Image):
                return ([Image().encode_example(value) if value is not None else None for value in data], Image())
        return (data, None)

    def __arrow_array__(self, type: Optional[pa.DataType]=None):
        """This function is called when calling pa.array(typed_sequence)"""
        if type is not None:
            raise ValueError('TypedSequence is supposed to be used with pa.array(typed_sequence, type=None)')
        del type
        data = self.data
        if self.type is None and self.try_type is None:
            data, self._inferred_type = self._infer_custom_type_and_encode(data)
        if self._inferred_type is None:
            type = self.try_type if self.trying_type else self.type
        else:
            type = self._inferred_type
        pa_type = get_nested_type(type) if type is not None else None
        optimized_int_pa_type = get_nested_type(self.optimized_int_type) if self.optimized_int_type is not None else None
        trying_cast_to_python_objects = False
        try:
            if isinstance(pa_type, _ArrayXDExtensionType):
                storage = to_pyarrow_listarray(data, pa_type)
                return pa.ExtensionArray.from_storage(pa_type, storage)
            if isinstance(data, np.ndarray):
                out = numpy_to_pyarrow_listarray(data)
            elif isinstance(data, list) and data and isinstance(first_non_null_value(data)[1], np.ndarray):
                out = list_of_np_array_to_pyarrow_listarray(data)
            else:
                trying_cast_to_python_objects = True
                out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
            if self.trying_int_optimization:
                if pa.types.is_int64(out.type):
                    out = out.cast(optimized_int_pa_type)
                elif pa.types.is_list(out.type):
                    if pa.types.is_int64(out.type.value_type):
                        out = array_cast(out, pa.list_(optimized_int_pa_type))
                    elif pa.types.is_list(out.type.value_type) and pa.types.is_int64(out.type.value_type.value_type):
                        out = array_cast(out, pa.list_(pa.list_(optimized_int_pa_type)))
            elif type is not None:
                out = cast_array_to_feature(out, type, allow_number_to_str=not self.trying_type)
            return out
        except (TypeError, pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError) as e:
            if not self.trying_type and isinstance(e, pa.lib.ArrowNotImplementedError):
                raise
            if self.trying_type:
                try:
                    if isinstance(data, np.ndarray):
                        return numpy_to_pyarrow_listarray(data)
                    elif isinstance(data, list) and data and any((isinstance(value, np.ndarray) for value in data)):
                        return list_of_np_array_to_pyarrow_listarray(data)
                    else:
                        trying_cast_to_python_objects = True
                        return pa.array(cast_to_python_objects(data, only_1d_for_numpy=True))
                except pa.lib.ArrowInvalid as e:
                    if 'overflow' in str(e):
                        raise OverflowError(f'There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})') from None
                    elif self.trying_int_optimization and 'not in range' in str(e):
                        optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                        logger.info(f'Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64.')
                        return out
                    elif trying_cast_to_python_objects and 'Could not convert' in str(e):
                        out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False))
                        if type is not None:
                            out = cast_array_to_feature(out, type, allow_number_to_str=True)
                        return out
                    else:
                        raise
            elif 'overflow' in str(e):
                raise OverflowError(f'There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})') from None
            elif self.trying_int_optimization and 'not in range' in str(e):
                optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                logger.info(f'Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64.')
                return out
            elif trying_cast_to_python_objects and 'Could not convert' in str(e):
                out = pa.array(cast_to_python_objects(data, only_1d_for_numpy=True, optimize_list_casting=False))
                if type is not None:
                    out = cast_array_to_feature(out, type, allow_number_to_str=True)
                return out
            else:
                raise