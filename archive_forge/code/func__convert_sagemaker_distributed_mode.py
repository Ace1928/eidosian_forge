import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_sagemaker_distributed_mode(value):
    value = int(value)
    return SageMakerDistributedType(['NO', 'DATA_PARALLEL', 'MODEL_PARALLEL'][value])