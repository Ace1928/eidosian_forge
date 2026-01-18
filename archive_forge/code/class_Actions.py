import enum
import threading
from cupyx.distributed import _klv_utils
class Actions(enum.IntEnum):
    Set = 1
    Get = 2
    Barrier = 3