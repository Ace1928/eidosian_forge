import argparse
import datetime
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import queue
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Tuple, Dict
from . import blas_compare_setup
def _subprocess_main(seed=0, num_threads=1, sub_label='N/A', result_file=None, env=None):
    import torch
    from torch.utils.benchmark import Timer
    conda_prefix = os.getenv('CONDA_PREFIX')
    assert conda_prefix
    if not torch.__file__.startswith(conda_prefix):
        raise ValueError(f'PyTorch mismatch: `import torch` resolved to `{torch.__file__}`, which is not in the correct conda env: {conda_prefix}')
    torch.manual_seed(seed)
    results = []
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 7, 96, 150, 225]:
        dtypes = (('Single', torch.float32), ('Double', torch.float64))
        shapes = (((n, n), (n, n), '(n x n) x (n x n)', 'Matrix-Matrix Product'), ((n, n), (n, 1), '(n x n) x (n x 1)', 'Matrix-Vector Product'))
        for (dtype_name, dtype), (x_shape, y_shape, shape_str, blas_type) in it.product(dtypes, shapes):
            t = Timer(stmt='torch.mm(x, y)', label=f'torch.mm {shape_str} {blas_type} ({dtype_name})', sub_label=sub_label, description=f'n = {n}', env=os.path.split(env or '')[1] or None, globals={'x': torch.rand(x_shape, dtype=dtype), 'y': torch.rand(y_shape, dtype=dtype)}, num_threads=num_threads).blocked_autorange(min_run_time=MIN_RUN_TIME)
            results.append(t)
    if result_file is not None:
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)