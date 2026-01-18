from ._base import issparse
from ._csr import csr_array
from ._sparsetools import csr_count_blocks
For a given blocksize=(r,c) count the number of occupied
    blocks in a sparse matrix A
    