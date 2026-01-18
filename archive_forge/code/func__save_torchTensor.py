import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def _save_torchTensor(pickler, obj):
    import torch

    def create_torchTensor(np_array):
        return torch.from_numpy(np_array)
    log(pickler, f'To: {obj}')
    args = (obj.detach().cpu().numpy(),)
    pickler.save_reduce(create_torchTensor, args, obj=obj)
    log(pickler, '# To')