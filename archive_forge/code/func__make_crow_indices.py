import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
@staticmethod
def _make_crow_indices(n_rows, n_cols, nnz, *, device, dtype, random=True):
    """Return crow_indices of a CSR tensor with size (n_rows, n_cols) and
        the number of specified elements nnz.

        If random is True, the column counts of rows are in random
        order. Otherwise, the column counts of rows are defined by the
        used sampling method.

        Sampling method
        ---------------

        The used sampling method was introduced in
        https://pearu.github.io/csr_sampling.html, and here we give
        only an overall description of the method.

        Notice that crow_indices can be defined as cumsum(counts)
        where counts is a sequence of non-negative integers satisfying
        the following conditions:

          len(counts) == n_rows + 1
          counts.max() <= n_cols

        while counts[i + 1] is interpreted as the number of specified
        elements in the i-th row.

        The used sampling method aims at increasing the diversity of
        CSR samples, that is, a CSR sample should contain (i) rows
        that are all filled, (ii) rows with no elements at all, and
        (iii) rows that are partially filled. At the same time and for
        the given total number of specified elements (nnz), there
        should be minimal preference to rows with a given number of
        elements.  To achieve this, the sampling method is built-up on
        using a sawteeth model for counts. In the simplest case, we
        would have

          counts = arange(n_rows + 1) % (n_cols + 1)

        that has equal number of all possible column counts per row.
        This formula can be used only for specific input values of
        n_rows, n_cols, and nnz. To generalize this model to any
        combinations of inputs, the counts model above is extended
        with an incomplete sawtooth, and the right and lower
        rectangular parts that will guarantee that

          counts.sum() == nnz

        for any combination of n_rows, n_cols, and nnz. Basically,
        we'll find a maximal window in (n_rows + 1, n_cols + 1)-grid
        that is able to hold a sequence of sawteeth and so-called
        final correction, while the external part of the window is
        filled with counts to meet the nnz constraint exactly.
        """
    assert 0 <= nnz <= n_rows * n_cols, (nnz, n_rows, n_cols)

    def sawteeth(n, m):
        M = (n_cols - m) * (n_cols - m + 1) // 2
        K = (n_rows - n) % (n_cols - m + 1)
        return M * ((n_rows - n) // (n_cols - m + 1)) + K * (K - 1) // 2
    counts = torch.zeros(n_rows + 1, dtype=dtype, device=torch.device('cpu'))
    n = m = 0
    N = sawteeth(n, m)
    if N and nnz >= max(N, n_cols):
        n_left = n
        n_right = n_rows - 1
        N_right = sawteeth(n_right, m)
        while n_right - n_left > 1:
            n_middle = (n_left + n_right) // 2
            N_middle = sawteeth(n_middle, m)
            if N_middle == 0 or nnz - n_middle * n_cols < max(N_middle, n_cols):
                n_right, N_right = (n_middle, N_middle)
            else:
                n_left = n_middle
        n, N = (n_right, N_right)
        assert n
        counts[-n:].fill_(n_cols)
    if N and nnz - n * n_cols >= max(N, n_rows - n):
        m_left = m
        m_right = n_cols - 1
        N_right = sawteeth(n, m_right)
        while m_right - m_left > 1:
            m_middle = (m_left + m_right) // 2
            N_middle = sawteeth(n, m_middle)
            if N_middle == 0 or nnz - n * n_cols - m_middle * (n_rows - n) < max(N_middle, n_rows - n):
                m_right, N_right = (m_middle, N_middle)
            else:
                m_left = m_middle
        m, N = (m_right, N_right)
        assert m
        counts[1:n_rows - n + 1].fill_(m)
    if N:
        q, r = divmod(nnz - n * n_cols - m * (n_rows - n), (n_cols - m) * (n_cols - m + 1) // 2)
        p = 1 + q * (n_cols - m + 1)
        k = math.isqrt(2 * r)
        if k * (k + 1) > 2 * r:
            k -= 1
        corr = r - k * (k + 1) // 2
        assert not (p > 1 and m > 0)
        counts[1:p] = torch.arange(p - 1, dtype=dtype, device=counts.device) % (n_cols - m + 1)
        counts[p:p + k + 1] += torch.arange(k + 1, dtype=dtype, device=counts.device)
    else:
        p = 1
        corr = nnz - n * n_cols - m * (n_rows - n)
    counts[p] += corr
    if random:
        perm = torch.randperm(n_rows, device=counts.device)
        counts[1:] = counts[1:][perm]
    crow_indices = counts
    crow_indices.cumsum_(dim=0)
    return crow_indices.to(device=device)