from enum import Enum, IntEnum
class TransformMethod(Enum):
    affine = 'transform'
    gcps = 'gcps'
    rpcs = 'rpcs'