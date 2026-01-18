import enum
import threading
from cupyx.distributed import _klv_utils
class ActionError:

    def __init__(self, exception):
        self._exception = exception

    def klv(self):
        e = self._exception
        return _klv_utils.get_result_action_t(1, str(e).encode('ascii'))

    @staticmethod
    def from_klv(klv):
        raise RuntimeError(klv._exception.decode('utf-8'))

    def decode_result(self, data):
        ActionError.from_klv(data)