from contextlib import contextmanager
class _NVTXStub:

    @staticmethod
    def _fail(*args, **kwargs):
        raise RuntimeError('NVTX functions not installed. Are you sure you have a CUDA build?')
    rangePushA = _fail
    rangePop = _fail
    markA = _fail