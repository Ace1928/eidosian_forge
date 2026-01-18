import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_compute_environment(value):
    value = int(value)
    return ComputeEnvironment(['LOCAL_MACHINE', 'AMAZON_SAGEMAKER'][value])