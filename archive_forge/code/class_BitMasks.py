import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
class BitMasks(enum.IntEnum):
    NONE = 0
    ALL = 255
    READ_ACCESS = 1
    READ_OFFSET = 3
    READ_CHUNK = 5
    READ_RANDOM = 7
    WRITE_ACCESS = 8
    WRITE_OFFSET = 24
    WRITE_CHUNK = 40
    WRITE_RANDOM = 56
    RW_ACCESS = 9
    RW_OFFSET = 27
    RW_CHUNK = 45
    RW_RANDOM = 63
    DRIVER_REUSABLE = 64