from contextlib import contextmanager
class _ITTStub:

    @staticmethod
    def _fail(*args, **kwargs):
        raise RuntimeError('ITT functions not installed. Are you sure you have a ITT build?')

    @staticmethod
    def is_available():
        return False
    rangePush = _fail
    rangePop = _fail
    mark = _fail