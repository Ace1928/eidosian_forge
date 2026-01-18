import flatbuffers
from flatbuffers.compat import import_numpy
class ModelOptimizationMode(object):
    PTQ_FLOAT16 = 1001
    PTQ_DYNAMIC_RANGE = 1002
    PTQ_FULL_INTEGER = 1003
    PTQ_INT16 = 1004
    QUANTIZATION_AWARE_TRAINING = 2000
    RANDOM_SPARSITY = 3001
    BLOCK_SPARSITY = 3002
    STRUCTURED_SPARSITY = 3003