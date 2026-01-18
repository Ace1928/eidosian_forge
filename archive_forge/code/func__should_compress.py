from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def _should_compress(num_rows, num_cols, matrix_approximation_rank, min_compression_rate):
    """
    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    including statistics describing the expected savings from compression.  We consider a tensor worth
    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where
    uncompressed size = ``num_rows`` * ``num_cols``,
    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.

    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:

    compression_recommendation is true if the tensor is worth compressing, and false otherwise (see above);

    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,

    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    """
    uncompressed_size = num_rows * num_cols
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (compressed_size * min_compression_rate < uncompressed_size, uncompressed_size, compressed_size)