import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def _validate_model_opset_version(self, model_opset_version: Optional[Dict[str, int]]):
    """Compare model_opset_version with expected_opset_version and raise error if we can't resolve the version
        difference.
        E.g., model_opset_version = {"aten": 3, "custom": 4}
        expected_opset_version = {"aten": 4, "custom": 4}
        This means we can use an upgrader for ATen to reconcile the deserialized model.

        The logic of this method:

        For common op namespaces:
        1. if model version < expected version, this case can be handled by upgraders.
        2. if model version > expected version, we need downgraders but not implemented yet.
        3. if model version == expected version, we don't need extra handling.

        For op namespace only in model_opset_version, we should give a warning because it is missing from
        expected_opset_version.
        """
    if not model_opset_version:
        raise RuntimeError('Serialized model should have opset version.')
    common_namespaces = {key for key in model_opset_version if key in self.expected_opset_version}
    for namespace in common_namespaces:
        assert isinstance((model_version := model_opset_version[namespace]), int), f'model_opset_version value should be int, got {model_opset_version[namespace]}'
        assert isinstance((compiler_version := self.expected_opset_version[namespace]), int), f'expected_opset_version value should be int, got {self.expected_opset_version[namespace]}'
        if model_version != compiler_version:
            raise NotImplementedError(f"Model opset version {model_opset_version} doesn't match to compiler opset version {self.expected_opset_version}! Upgrader/downgrader is not implemented yet.")
    for namespace in model_opset_version:
        if namespace in common_namespaces:
            continue
        log.warning("Compiler doesn't have a version table for op namespace: {ns}. ", extra={'ns': namespace})