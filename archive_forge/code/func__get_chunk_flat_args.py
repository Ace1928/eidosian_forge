from torch._functorch.vmap import (vmap_impl, _check_randomness_arg,
from torch._functorch.utils import exposed_in, argnums_t
import functools
def _get_chunk_flat_args(flat_args_, flat_in_dims_, chunks_):
    flat_args_chunks = tuple((t.chunk(chunks_, dim=in_dim) if in_dim is not None else [t] * chunks_ for t, in_dim in zip(flat_args_, flat_in_dims_)))
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args