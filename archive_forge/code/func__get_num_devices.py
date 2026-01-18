import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
def _get_num_devices(self):
    warnings.warn('Caution using SyncBatchNorm: if not using all the GPUs, please mannually set num_devices', UserWarning)
    num_devices = context.num_gpus()
    num_devices = num_devices if num_devices > 0 else 1
    return num_devices