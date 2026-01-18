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
class ExportedProgramDeserializer:

    def __init__(self, expected_opset_version: Optional[Dict[str, int]]=None):
        self.expected_opset_version: Dict[str, int] = {}
        if expected_opset_version:
            self.expected_opset_version.update(expected_opset_version)
        if 'aten' not in self.expected_opset_version:
            self.expected_opset_version['aten'] = torch._C._get_max_operator_version()

    def deserialize_range_constraints(self, symbol_name_to_range: Dict[str, symbolic_shapes.ValueRanges], symbol_name_to_symbol: Dict[str, sympy.Symbol]) -> Dict[sympy.Symbol, ValueRanges]:
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if (symbol := symbol_name_to_symbol.get(k)):
                range_constraints[symbol] = v
            else:
                log.warning(f'Symbol {k} did not appear in the graph that was deserialized')
        return range_constraints

    def deserialize(self, serialized_artifact: SerializedArtifact) -> ep.ExportedProgram:
        assert isinstance(serialized_artifact.exported_program, ExportedProgram)
        if serialized_artifact.exported_program.schema_version != SCHEMA_VERSION:
            raise SerializeError(f'Serialized schema version {serialized_artifact.exported_program.schema_version} does not match our current schema version {SCHEMA_VERSION}.')
        symbol_name_to_range = {k: symbolic_shapes.ValueRanges(_int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val)) for k, v in serialized_artifact.exported_program.range_constraints.items()}
        constants = deserialize_torch_artifact(serialized_artifact.constants)
        tensor_constants = {k: v for k, v in constants.items() if isinstance(v, torch.Tensor)}
        res = GraphModuleDeserializer().deserialize(serialized_artifact.exported_program.graph_module, symbol_name_to_range, constants)
        range_constraints = self.deserialize_range_constraints(symbol_name_to_range, res.names_to_symbols)
        model_opset_version: Optional[Dict[str, int]] = serialized_artifact.exported_program.opset_version
        self._validate_model_opset_version(model_opset_version)
        upgrader = GraphModuleOpUpgrader(self.expected_opset_version, model_opset_version)
        state_dict = deserialize_torch_artifact(serialized_artifact.state_dict)
        exported_program = ep.ExportedProgram(res.graph_module, res.graph_module.graph, res.signature, state_dict, range_constraints, [], res.module_call_graph, None, load_verifier(serialized_artifact.exported_program.dialect), tensor_constants=tensor_constants)
        return upgrader.upgrade(exported_program)

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