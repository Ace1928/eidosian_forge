import enum
import threading
from cupyx.distributed import _klv_utils
class GetResult:

    def __init__(self, value):
        self.value = value

    def klv(self):
        v = _klv_utils.create_value_bytes(self.value)
        action = _klv_utils.get_result_action_t(0, v)
        return bytes(action)

    @staticmethod
    def from_klv(value):
        value = bytearray(value)
        return _klv_utils.get_value_from_bytes(value)