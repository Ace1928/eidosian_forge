import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
@enum.unique
class Compression(enum.IntEnum):
    """Indicates the compression method to be used for an RPC.

    Attributes:
     NoCompression: Do not use compression algorithm.
     Deflate: Use "Deflate" compression algorithm.
     Gzip: Use "Gzip" compression algorithm.
    """
    NoCompression = _compression.NoCompression
    Deflate = _compression.Deflate
    Gzip = _compression.Gzip