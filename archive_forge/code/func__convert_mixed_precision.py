import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_mixed_precision(value):
    value = int(value)
    return PrecisionType(['no', 'fp16', 'bf16', 'fp8'][value])