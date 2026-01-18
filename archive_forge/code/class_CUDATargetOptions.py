from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext
class CUDATargetOptions(TargetOptions):
    pass