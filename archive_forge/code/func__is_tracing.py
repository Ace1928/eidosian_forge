import os
import warnings
from modulefinder import Module
import torch
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
def _is_tracing():
    return torch._C._get_tracing_state()