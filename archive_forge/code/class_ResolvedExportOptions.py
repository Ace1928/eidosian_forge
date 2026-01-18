from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
class ResolvedExportOptions(ExportOptions):
    """Consolidates :class:`ExportOptions` with default values.
    All unspecified options from :class:`ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """
    dynamic_shapes: bool
    op_level_debug: bool
    diagnostic_options: DiagnosticOptions
    fake_context: ONNXFakeContext
    onnx_registry: OnnxRegistry
    decomposition_table: Dict[torch._ops.OpOverload, Callable]
    'A dictionary that maps operators to their decomposition functions.'
    onnxfunction_dispatcher: torch.onnx._internal.fx.onnxfunction_dispatcher.OnnxFunctionDispatcher
    'The ONNX dispatcher used to dispatch ATen operators to ONNX functions.'
    fx_tracer: FXGraphExtractor
    'The FXGraphExtractor instance used to extract the FX graph from the model.'
    diagnostic_context: diagnostics.DiagnosticContext
    'The diagnostics context for the export. Responsible for recording diagnostics,\n    logging diagnostics, and generating the SARIF log.'

    @_beartype.beartype
    def __init__(self, options: Union[ExportOptions, 'ResolvedExportOptions'], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None):
        from torch.onnx._internal.fx import diagnostics, dynamo_graph_extractor, torch_export_graph_extractor
        if isinstance(options, ResolvedExportOptions):
            self.dynamic_shapes = options.dynamic_shapes
            self.op_level_debug = options.op_level_debug
            self.diagnostic_options = options.diagnostic_options
            self.fake_context = options.fake_context
            if isinstance(model, torch_export.ExportedProgram) and (not isinstance(options.fx_tracer, torch_export_graph_extractor.TorchExport)):
                message = "'model' of type 'ExportedProgram' is only supported with 'TorchExport' FX Tracer"
                e = InvalidExportOptionsError(message)
                raise InvalidExportOptionsError(ONNXProgram._from_failure(e, options.diagnostic_context), message)
            self.fx_tracer = options.fx_tracer
            self.onnx_registry = options.onnx_registry
            self.onnxfunction_dispatcher = options.onnxfunction_dispatcher
            self.decomposition_table = options.decomposition_table
            self.diagnostic_context = options.diagnostic_context
        else:
            T = TypeVar('T')

            @_beartype.beartype
            def resolve(value: Optional[T], fallback: Union[T, Callable[[], T]]) -> T:
                if value is not None:
                    return value
                if callable(fallback):
                    return fallback()
                return fallback
            self.dynamic_shapes = resolve(options.dynamic_shapes, False)
            self.diagnostic_options = resolve(options.diagnostic_options, DiagnosticOptions())
            if isinstance(model, torch_export.ExportedProgram):
                self.fx_tracer = torch_export_graph_extractor.TorchExport()
            else:
                self.fx_tracer = dynamo_graph_extractor.DynamoExport()
            self.fake_context = resolve(options.fake_context, None)
            self.diagnostic_context = diagnostics.DiagnosticContext('torch.onnx.dynamo_export', torch.__version__, self.diagnostic_options)
            self.onnx_registry = resolve(options.onnx_registry, OnnxRegistry())
            self.decomposition_table = decomposition_table.create_onnx_friendly_decomposition_table(self.onnx_registry)
            from torch.onnx._internal.fx import onnxfunction_dispatcher
            self.op_level_debug = resolve(options.op_level_debug, False)
            self.onnxfunction_dispatcher = onnxfunction_dispatcher.OnnxFunctionDispatcher(self.onnx_registry, self.diagnostic_context)
            for key in dir(options):
                if not key.startswith('_'):
                    assert hasattr(self, key), f"Unresolved option '{key}'"