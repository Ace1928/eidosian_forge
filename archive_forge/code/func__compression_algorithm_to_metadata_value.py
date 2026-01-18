from __future__ import annotations
from typing import Optional
import grpc
from grpc._cython import cygrpc
from grpc._typing import MetadataType
def _compression_algorithm_to_metadata_value(compression: grpc.Compression) -> str:
    return _METADATA_STRING_MAPPING[compression]