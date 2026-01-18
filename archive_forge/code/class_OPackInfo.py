from gitdb.util import bin_to_hex
from gitdb.fun import (
class OPackInfo(tuple):
    """As OInfo, but provides a type_id property to retrieve the numerical type id, and
    does not include a sha.

    Additionally, the pack_offset is the absolute offset into the packfile at which
    all object information is located. The data_offset property points to the absolute
    location in the pack at which that actual data stream can be found."""
    __slots__ = tuple()

    def __new__(cls, packoffset, type, size):
        return tuple.__new__(cls, (packoffset, type, size))

    def __init__(self, *args):
        tuple.__init__(self)

    @property
    def pack_offset(self):
        return self[0]

    @property
    def type(self):
        return type_id_to_type_map[self[1]]

    @property
    def type_id(self):
        return self[1]

    @property
    def size(self):
        return self[2]