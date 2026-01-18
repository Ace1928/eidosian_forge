import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
class DynamoBackend(str, BaseEnum):
    """
    Represents a dynamo backend (see https://pytorch.org/docs/stable/torch.compiler.html).

    Values:

        - **NO** -- Do not use torch dynamo.
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **AOT_TS_NVFUSER** -- nvFuser with AotAutograd/TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **NVPRIMS_NVFUSER** -- nvFuser with PrimTorch. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **TENSORRT** -- Uses ONNXRT to run TensorRT for inference optimizations. [Read
          more](https://github.com/onnx/onnx-tensorrt)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).
        - **TVM** -- Uses Apach TVM for inference optimizations. [Read more](https://tvm.apache.org/)

    """
    NO = 'NO'
    EAGER = 'EAGER'
    AOT_EAGER = 'AOT_EAGER'
    INDUCTOR = 'INDUCTOR'
    AOT_TS_NVFUSER = 'AOT_TS_NVFUSER'
    NVPRIMS_NVFUSER = 'NVPRIMS_NVFUSER'
    CUDAGRAPHS = 'CUDAGRAPHS'
    OFI = 'OFI'
    FX2TRT = 'FX2TRT'
    ONNXRT = 'ONNXRT'
    TENSORRT = 'TENSORRT'
    IPEX = 'IPEX'
    TVM = 'TVM'