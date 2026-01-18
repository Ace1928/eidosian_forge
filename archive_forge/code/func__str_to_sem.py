from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _str_to_sem(sem_option):
    sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
    if sem_option:
        if sem_option == 'acquire':
            sem = ir.MEM_SEMANTIC.ACQUIRE
        elif sem_option == 'release':
            sem = ir.MEM_SEMANTIC.RELEASE
        elif sem_option == 'acq_rel':
            sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
        elif sem_option == 'relaxed':
            sem = ir.MEM_SEMANTIC.RELAXED
        else:
            raise ValueError(f'Memory semantic {sem_option} not supported')
    return sem