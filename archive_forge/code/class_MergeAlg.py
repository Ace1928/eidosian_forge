from enum import Enum, IntEnum
class MergeAlg(Enum):
    """Available rasterization algorithms"""
    replace = 'REPLACE'
    add = 'ADD'