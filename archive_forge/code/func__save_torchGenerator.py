import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def _save_torchGenerator(pickler, obj):
    import torch

    def create_torchGenerator(state):
        generator = torch.Generator()
        generator.set_state(state)
        return generator
    log(pickler, f'Ge: {obj}')
    args = (obj.get_state(),)
    pickler.save_reduce(create_torchGenerator, args, obj=obj)
    log(pickler, '# Ge')