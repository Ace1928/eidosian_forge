from datetime import timedelta
import logging
import os
import threading
import warnings
from typing import Generator, Tuple
from urllib.parse import urlparse
import torch
import torch.distributed as dist
def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
    type_mapping = {backend: backend_registry.BackendType, store: dist.Store, name: str, rank: numbers.Integral, world_size: (numbers.Integral, type(None)), rpc_backend_options: RpcBackendOptions}
    for arg, arg_type in type_mapping.items():
        if not isinstance(arg, arg_type):
            raise RuntimeError(f'Argument {arg} must be of type {arg_type} but got type {type(arg)}')