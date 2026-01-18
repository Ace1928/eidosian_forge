from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor
class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """

    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self.submit(func, *args)
        future.get = future.result
        return future

    def map(self, f, *args):
        return list(super().map(f, *args))