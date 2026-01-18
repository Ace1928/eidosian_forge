import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch
import sympy
import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel

        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller
        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        