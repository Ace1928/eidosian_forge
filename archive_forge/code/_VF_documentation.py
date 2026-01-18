import sys
import types
import torch

This makes the functions in torch._C._VariableFunctions available as
    torch._VF.<funcname>
without mypy being able to find them.

A subset of those functions are mapped to ATen functions in
torch/jit/_builtins.py

See https://github.com/pytorch/pytorch/issues/21478 for the reason for
introducing torch._VF

