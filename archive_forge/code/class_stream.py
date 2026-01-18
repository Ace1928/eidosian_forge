from contextlib import contextmanager
from .cudadrv.devices import require_context, reset, gpus  # noqa: F401
from .kernel import FakeCUDAKernel
from numba.core.sigutils import is_signature
from warnings import warn
from ..args import In, Out, InOut  # noqa: F401
class stream(object):
    """
    The stream API is supported in the simulator - however, all execution
    occurs synchronously, so synchronization requires no operation.
    """

    @contextmanager
    def auto_synchronize(self):
        yield

    def synchronize(self):
        pass