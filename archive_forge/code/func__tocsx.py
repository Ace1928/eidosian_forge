import cupy
from cupyx import cusparse
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
def _tocsx(self):
    """Inverts the format.
        """
    return self.tocsr()