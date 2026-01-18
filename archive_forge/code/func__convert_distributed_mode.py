import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_distributed_mode(value):
    value = int(value)
    return DistributedType(['NO', 'MULTI_CPU', 'MULTI_XPU', 'MULTI_GPU', 'MULTI_NPU', 'XLA'][value])