from enum import Enum, IntEnum
class MaskFlags(IntEnum):
    all_valid = 1
    per_dataset = 2
    alpha = 4
    nodata = 8