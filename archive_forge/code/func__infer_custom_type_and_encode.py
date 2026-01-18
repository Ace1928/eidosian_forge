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